import mido
import numpy as np

def midi2bytes(filename, min_note_denom=32):
    PITCHES = 12 # const for number of pitches in octave
    OCTAVES = 10 # const for number of octavies in midi

    midifile = mido.MidiFile(filename)

    if min_note_denom % 4 != 0:
        raise ValueError("min_note_denom is not divisible by 4")

    if midifile.ticks_per_beat % (min_note_denom // 4) != 0:
        print("Warning: call to midi2bytes with min_note_denom " + str(min_note_denom) \
            + ", but those notes do not have an integer delta time")
    delta_step = midifile.ticks_per_beat // (min_note_denom // 4)

    # create array for notes and array for times
    times = np.zeros((0), dtype=np.int64)
    for track in midifile.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == 'note_on' or msg.type == 'note_off':
                times = np.append(times, t)
            else:
                pass

    times //= delta_step
    times = np.unique(times)

    b = np.zeros((times.size, PITCHES, OCTAVES), dtype=np.int8)

    # put in the notes
    for track in midifile.tracks:
        t = 0
        t_prev = 0
        notes = np.zeros((PITCHES, OCTAVES), dtype=np.int8)
        for msg in track:
            t += msg.time
            
            # write notes as soon as the time changes
            if t_prev != t:
                t_prev = t
                b[ti,:,:] = np.maximum(notes, b[ti,:,:])

            ti = np.where(times == t // delta_step)[0]
            if (msg.type == 'note_on' and msg.velocity == 0) or msg.type == 'note_off':
                notes[msg.note % PITCHES, msg.note // PITCHES] = 0
            elif msg.type == 'note_on':
                notes[msg.note % PITCHES, msg.note // PITCHES] = 127

        # write notes at the end of the track
        b[ti,:,:] = np.maximum(notes, b[ti,:,:])

    return times, b

def bytes2midi(times, b, min_note_denom=32):
    PITCHES = 12 # const for number of pitches in octave
    OCTAVES = 10 # const for number of octavies in midi

    midifile = mido.MidiFile()
    midifile.add_track()

    if min_note_denom % 4 != 0:
        raise ValueError("min_note_denom is not divisible by 4")

    if midifile.ticks_per_beat % (min_note_denom // 4) != 0:
        print("Warning: call to midi2bytes with delta_step = None, but sixteenth notes do not\
            have an integer delta time")
    delta_step = midifile.ticks_per_beat // (min_note_denom // 4)

    # set sustain pedal
    midifile.tracks[0].append(mido.Message('control_change', control=64, value=127))
    # set reverb level
    midifile.tracks[0].append(mido.Message('control_change', control=91, value=64))

    times *= delta_step

    missedTime = 0

    # put in the notes
    for i in range(times.size):
        if i > 0:
            notesDiff = np.where(b[i,:,:] != b[i-1,:,:])
        else:
            notesDiff = np.where(b[i,:,:] != 0)
        if i > 0:
            dtime = times[i] - times[i-1]
        else:
            dtime = times[i]
        
        numNote = 0
        
        if notesDiff[0].size == 0:
            missedTime += dtime
        else:
            for j in range(0,notesDiff[0].size):
                pitch = notesDiff[0][j]
                octave = notesDiff[1][j]
                note = PITCHES * octave + pitch

                time = dtime + missedTime if numNote == 0 else 0
                vel = b[i,notesDiff[0][j],notesDiff[1][j]]
                msg = mido.Message('note_on', note=note, time=time, velocity=vel)
                midifile.tracks[0].append(msg)
                numNote += 1 
            missedTime = 0

    return midifile
