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

    # calculate repetitions of notes
    rep = np.zeros((t.size, 1))
    rep[0,0] = t[0]
    rep[1:,0] = np.diff(t)

    # append time steps to data
    data = np.hstack((rep, notes))

    return data

def postprocess(data):
    rep, notes = np.split(data, [1], axis=1)

    # turn notes into integers from 0 to 127
    notes = notes * 127
    notes = notes.astype(np.int8, copy=False)

    # turn repetitions into times
    t = np.zeros((rep.size), np.int64)
    t[0] = rep[0]
    for i in range(1, t.size):
        t[i] = t[i-1] + rep[i]

    return bytes2midi(t, notes)

if __name__ == '__main__':
    data = np.zeros((0, 129), dtype=np.float32)
    for f in args.files:
        datai = preprocess(f)
        pad = np.zeros((1, 129), dtype=np.float32)
        pad[0,0] = args.waittime

        data = np.vstack((data, pad, datai))
    
    print("Created data with")
    print("\t%s songs"%(len(args.files)))
    print("\t%s timesteps"%(data.shape[0]))
    print("\t%s 32nd notes"%(np.sum(data[:,0])))

    if args.midi:
        mfile = postprocess(data)
        mfile.save(args.outputfile)
    else:
        np.save(args.outputfile, data)
