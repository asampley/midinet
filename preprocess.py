from midi2bytes import *
import numpy as np
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess midi files for the neural network by turning them into vectors of notes')
    parser.add_argument('outputfile', help='File to save the numpy array to')
    parser.add_argument('files', nargs='+', help='Midi files to turn into data')
    parser.add_argument('--waittime', '-t', help='Number of 32nd notes to wait in between songs', type=int, default=32) # default 1 whole note
    parser.add_argument('--midi', '-m', help='Save as a midi file instead of a numpy file', action='store_true')
    args = parser.parse_args()

def preprocess(f):
    t, notes = midi2bytes(f)

    # turn notes into reals from zero to one
    notes.astype(np.float32, copy=False)
    notes = notes / 127

    # calculate repetitions of notes rather than timestamps
    rep = np.zeros((t.size,))
    rep[0] = t[0]
    rep[1:] = np.diff(t)

    return rep, notes

def postprocess(rep, notes):
    # turn notes into integers from 0 to 127
    notes = notes * 127
    notes = notes.astype(np.int8, copy=False)

    # turn repetitions into times
    t = np.zeros((rep.size,), np.int64)
    t[0] = rep[0]
    for i in range(1, t.size):
        t[i] = t[i-1] + rep[i]

    return bytes2midi(t, notes)

if __name__ == '__main__':
    PITCHES = 12 # pitches in an octave
    OCTAVES = 10 # octaves in midi
    notes = np.zeros((0, PITCHES, OCTAVES), dtype=np.float32)
    reps  = np.zeros((0,), dtype=np.float32)
    
    # create padding between songs
    pad = np.zeros((1, PITCHES, OCTAVES), dtype=np.float32)
    padreps = np.array((args.waittime,))
    for f in args.files:
        repsi, notesi = preprocess(f)

        reps = np.concatenate((reps, padreps, repsi), axis=0)
        notes = np.concatenate((notes, pad, notesi), axis=0)
    
    assert(reps.shape[0] == notes.shape[0], "Programming error, repetitions and notes do not have the same number of elements")

    print("Created data with")
    print("\t%s songs"%(len(args.files)))
    print("\t%s timesteps"%(reps.shape[0]))
    print("\t%s 32nd notes"%(np.sum(reps[:])))

    if args.midi:
        mfile = postprocess(reps, notes)
        mfile.save(args.outputfile)
    else:
        np.savez(args.outputfile, reps=reps, notes=notes)
