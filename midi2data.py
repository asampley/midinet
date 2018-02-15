import mido
import numpy as np
import math

_PITCHES = 12
_OCTAVES = 11

def velocity2volume(velocity, volumes):
    """
    Returns a categorical volume.

    0 if the velocity is 0
    [1, volumes) if the velocity is not 0, by equally dividing the interval [0,128) into <volumes-1> sections

    Output is undefined for messages that are not note_on
    """
    return 1 + (volumes-1) * velocity // 128 if velocity != 0 else 0

def volume2velocity(volume, volumes):
    """
    Returns a velocity for a note based on the categorical volume.
    Can be approximated as the inverse of _message2volume, 
    but it only returns the highest possible velocity that would produce the volume
    """
    return volume * 128 // (volumes-1) - 1 if volume != 0 else 0

def note2pitch_octave(note):
    """
    Returns a pitch and octave for a note
    """
    return note % _PITCHES, note // 12

def pitch_octave2note(pitch, octave):
    """
    Returns a midi note for a pitch and octave
    """
    return _PITCHES * octave + pitch

def time2ticks(time, time_step):
    """
    Returns a number of ticks based on the time_step
    The ticks are always rounded up, there is not an exact number of time_steps in the time
    """
    return math.ceil(time / time_step)

def ticks2time(ticks, time_step):
    """
    Returns the time corresponding to a number of ticks
    """
    return ticks * time_step

def ticks2durations(ticks, duration_categories):
    """
    Returns a list of durations corresponding to a number of ticks
    If there are more ticks than can be represented by a single duration, the list will
    have a length greater than 1. The durations will be sorted in descending order.
    """
    assert(ticks >= 0)
    assert(duration_categories >= 2)

    if ticks == 0:
        return [0]

    durations = []

    remaining_ticks = ticks
    note_ticks = 2 ** (duration_categories - 2)
    duration = duration_categories - 1
    while remaining_ticks > 0:
        if note_ticks <= remaining_ticks:
            remaining_ticks -= note_ticks
            durations += [duration]
        else:
            duration -= 1
            note_ticks //= 2
    return durations

def durations2ticks(durations):
    """
    Converts a list of durations back into a single value of ticks
    """
    ticks = 0
    for duration in durations:
        ticks += 2 ** (duration - 1) if duration != 0 else 0
    return ticks

def midi2data(filename, min_note_denom=32, duration_categories=8, volumes=2):
    DURATIONS = duration_categories # number of duration categories, from 0 notes, to (2^(DURATIONS-2))/min_note_denom notes. Must be at least 2.
    VOLUMES   = volumes             # number of volumes, evenly dividing 0 to 127. Must be at least 2.

    midifile = mido.MidiFile(filename)

    if min_note_denom % 4 != 0:
        raise ValueError("min_note_denom is not divisible by 4")

    if midifile.ticks_per_beat % (min_note_denom // 4) != 0:
        print("Warning: call to midi2data with min_note_denom " + str(min_note_denom) \
            + ", but those notes do not have an integer delta time")
    delta_step = midifile.ticks_per_beat // (min_note_denom // 4)

    # create array with the following four elements in each row
    # [pitch, octave, volume, duration]
    messages = []
    for msg in mido.merge_tracks(midifile.tracks):
        if msg.is_meta or not (msg.type == 'note_on' or msg.type == 'note_off'):
            continue
        
        # convert message note to pitch and octave
        pitch, octave = note2pitch_octave(msg.note)

        # convert message velocity into volume.
        volume = 0 if msg.type == 'note_off' else velocity2volume(msg.velocity, VOLUMES)

        # convert message time into several messages that add up to the total time, within delta_step
        ticks     = time2ticks(msg.time, delta_step)
        durations = ticks2durations(ticks, DURATIONS)
        for duration in durations:
            messages += [[pitch, octave, volume, duration]]

    # finally, turn it into a numpy array, and return it
    messages = np.array(messages)
    return messages

def data2midi(messages, min_note_denom=32, duration_categories=8, volumes=2):
    assert(messages.shape[1] == 4 and np.issubdtype(messages.dtype, np.integer))
    
    DURATIONS = duration_categories # number of duration categories, from 0 notes, to (2^(DURATIONS-1))/min_note_denom notes. Must be at least 2.
    VOLUMES   = volumes             # number of volumes, evenly dividing 0 to 127. Must be at least 2.

    midifile = mido.MidiFile()
    midifile.add_track()
    track = midifile.tracks[0]

    if min_note_denom % 4 != 0:
        raise ValueError("min_note_denom is not divisible by 4")

    if midifile.ticks_per_beat % (min_note_denom // 4) != 0:
        print("Warning: call to data2midi with min_note_denom " + str(min_note_denom) \
            + ", but those notes do not have an integer delta time")
    delta_step = midifile.ticks_per_beat // (min_note_denom // 4)

    # set sustain pedal
    track.append(mido.Message('control_change', control=64, value=127))
    # set reverb level
    track.append(mido.Message('control_change', control=91, value=64))

    i = 0
    durations = []
    while i < messages.shape[0]:
        # append the duration of the message onto the end
        durations += [messages[i,3]]

        # if any of the message (excluding the duration) is different from the next, put all the accumulated durations onto the midi file
        # also do this on the last message
        if i == messages.shape[0] - 1 or not np.all(messages[i,:3] == messages[i+1,:3]):
            note = pitch_octave2note(messages[i,0], messages[i,1])
            time = ticks2time(durations2ticks(durations), delta_step)
            velocity = volume2velocity(messages[i,2], VOLUMES)

            msg = mido.Message('note_on', note=note, time=time, velocity=velocity)
            track.append(msg)
            durations = []
        
        # increment counter
        i += 1

    return midifile
