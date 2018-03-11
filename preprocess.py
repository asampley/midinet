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
    parser.add_argument('--volumes', '-v', help='Number of different volumes to save', type=int, default=1)
    parser.add_argument('--durations', '-d', help='Number of different durations to save', type=int, default=8)
    args = parser.parse_args()

def preprocess(files, volumes=1, duration_categories=8, pad_duration=0):

    # create padding between songs
    pad_durations = m2d.ticks2durations(pad_duration, duration_categories)
    pad = np.zeros((len(pad_durations), 4), dtype=np.int8)
    pad[:,3] = pad_durations
    
    # go through each file and add them to the list with pads in between
    message_arrays = []

    for f in files:
        messages_f = m2d.midi2data(f, volumes=volumes, duration_categories=duration_categories)
        message_arrays += [messages_f, pad]

    # turn list into unique messages, and indices, s.t. messages[indices] gives the original song
    messages = np.concatenate(message_arrays, axis=0)
    messages, indices = np.unique(messages, return_inverse=True, axis=0)

    return messages, indices

def postprocess(messages, indices, volumes=1, duration_categories=8):
    return m2d.data2midi(messages[indices], volumes=volumes, duration_categories=duration_categories)

if __name__ == '__main__':

    # preprocess
    messages, indices = preprocess(args.files, volumes=args.volumes, duration_categories=args.durations, pad_duration=args.waittime)
    names = ['pitch', 'octave', 'volume', 'duration']
    maxes = [m2d._PITCHES, m2d._OCTAVES, args.volumes, args.durations]

    print("Created data with")
    print("\t%s songs"%(len(args.files)))
    print("\t%s unique messages"%(messages.shape[0]))
    print("\t%s timesteps"%(indices.shape[0]))

    if args.midi:
        mfile = postprocess(messages, indices, volumes=args.volumes, duration_categories=args.durations)
        mfile.save(args.outputfile)
    else:
        np.savez(args.outputfile, messages=messages, indices=indices, names=names, maxes=maxes)
