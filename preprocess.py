import midi2data as m2d
import numpy as np
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess midi files for the neural network by turning them into tensors describing notes')
    parser.add_argument('outputfile', help='File to save the numpy array to')
    parser.add_argument('files', nargs='+', help='Midi files to turn into data')
    parser.add_argument('--waittime', '-t', help='Number of 32nd notes to wait in between songs', type=int, default=32) # default 1 whole note
    parser.add_argument('--midi', '-m', help='Save as a midi file instead of a numpy file', action='store_true')
    parser.add_argument('--volumes', '-v', help='Number of different volumes to save', type=int, default=2)
    parser.add_argument('--durations', '-d', help='Number of different durations to save', type=int, default=8)
    args = parser.parse_args()

def preprocess(f, volumes=2, duration_categories=8):
    return m2d.midi2data(f, volumes=volumes, duration_categories=duration_categories)

def postprocess(messages, volumes=2, duration_categories=8):
    # trim invalid midi notes
    messages = messages[np.logical_or(messages[:,0] < 8, messages[:,1] != 10),:]
    return m2d.data2midi(messages, volumes=volumes, duration_categories=duration_categories)

if __name__ == '__main__':
    # hold each output in a list
    message_arrays = []

    # create padding between songs
    pad_durations = m2d.ticks2durations(args.waittime, args.durations)
    pad = np.zeros((len(pad_durations), 4), dtype=np.int8)
    pad[:,3] = pad_durations
        
    # go through each file and add them to the list with pads in between
    for f in args.files:
        messages_f = preprocess(f, volumes=args.volumes, duration_categories=args.durations)
        message_arrays += [messages_f, pad]

    # concatenate all arrays into numpy array
    messages = np.concatenate(message_arrays, axis=0)

    print("Created data with")
    print("\t%s songs"%(len(args.files)))
    print("\t%s timesteps"%(messages.shape[0]))
    print("\t%s 32nd notes"%(np.sum(messages[:,3])))

    if args.midi:
        mfile = postprocess(messages)
        mfile.save(args.outputfile)
    else:
        maxes = (m2d._PITCHES, m2d._OCTAVES, args.volumes, args.durations)
        names = ('pitch', 'octave', 'volume', 'duration')
        np.savez(args.outputfile, messages=messages, maxes=maxes, names=names)
