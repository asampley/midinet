import mido
import numpy as np
import math

_PITCHES = 12
_OCTAVES = 11

def velocity2volume(velocity, volumes):
    """
    Returns a categorical volume.

    [0, volumes) is returned by equally dividing the velocity interval [0,128) into <volumes-1> sections

    Output is undefined for messages that are not note_on
    """
    return (volumes * velocity) // 128

def volume2velocity(volume, volumes):
    """
    Returns a velocity for a note based on the categorical volume.
    Can be approximated as the inverse of _message2volume, 
    but it only returns the highest possible velocity that would produce the volume
    """
    return ((volume + 1) * 128) // volumes - 1

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

def midi2data(filename, min_note_denom=32, duration_categories=8, volumes=1):
    """
    Convert a midi file into an array of 4-vectors, containing
    [pitch, octave, volume, duration]

    All note_off and note_on with velocity 0 messages are ignored.
    This means that notes cannot really be held and preserved through this function.
    """

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
    time = 0
    for msg in mido.merge_tracks(midifile.tracks):
        time += msg.time # accumulate time even for ignored messages
        if msg.is_meta or msg.type != 'note_on' or msg.velocity == 0:
            continue
        
        # convert message note to pitch and octave
        pitch, octave = note2pitch_octave(msg.note)

        # convert message velocity into volume.
        volume = velocity2volume(msg.velocity, volumes)

        # convert message time into several messages that add up to the total time, within delta_step
        ticks     = time2ticks(time, delta_step)
        durations = ticks2durations(ticks, duration_categories)
        for duration in durations:
            messages += [[pitch, octave, volume, duration]]
        
        time = 0 # reset time after adding a message

    # finally, turn it into a numpy array, and return it
    messages = np.array(messages)
    return messages

def data2midi(messages, min_note_denom=32, duration_categories=8, volumes=1):
    """
    Convert a numpy array of shape (?,4) into a midi file
    Notes are immediately followed by a note off signal, so notes cannot be held.
    """
    assert(messages.shape[1] == 4 and np.issubdtype(messages.dtype, np.integer))

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
        # as well as a note_off message immediately after
        # also do this on the last message
        if i == messages.shape[0] - 1 or not np.all(messages[i,:3] == messages[i+1,:3]):
            note = pitch_octave2note(messages[i,0], messages[i,1])
            time = ticks2time(durations2ticks(durations), delta_step)
            velocity = volume2velocity(messages[i,2], volumes)

            msg = mido.Message('note_on', note=note, time=time, velocity=velocity)
            track.append(msg)
            msg = mido.Message('note_off', note=note, time=0)
            track.append(msg)
            durations = []
        
        # increment counter
        i += 1

    return midifile
