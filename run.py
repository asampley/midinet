# Credit to https://gist.github.com/siemanko/b18ce332bde37e156034e5d3f60f8a23 for initial code

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import model
import random
from preprocess import postprocess
import os
import argparse
from datetime import datetime



# parse arguments
parser = argparse.ArgumentParser(description='Train and generate music with the neural network described in model.py')
parser.add_argument('--data', '-d', help='Numpy file containing the training data. Default "data/all.npz"', type=str, default='data/all.npz')
parser.add_argument('--epochs', '-n', help='Number of times to train and then generate. Default 1.', type=int, default=1)
parser.add_argument('--train', '-t', help='Set the number of batches for each epoch of training. Default 0.', type=int, default=0)
parser.add_argument('--generate', '-g', help='Generate a song every g epochs. Default 1.', type=int, default=1)
parser.add_argument('--savedir', '-s', help='Directory in which to save the neural network model. Default "model/".', default='model/')
parser.add_argument('--songlength', '-l', help='Length of song to generate. Default 100.', type=int, default=100)
parser.add_argument('--songprefix', '-p', help='Prefix to prepend to saved song files. Default "songs/".', type=str, default='songs/')
parser.add_argument('--rnns', help='Sizes of RNN layers. Default 256 512 256.', type=int, nargs='+', default=[256, 512, 256])
parser.add_argument('--denses', help='Sizes of Dense layers. Default 256.', type=int, nargs='+', default=[256])
group = parser.add_mutually_exclusive_group()
group.add_argument('--notemax', help='Select the note by taking the argmax of the neural network''s output', action='store_true')
group.add_argument('--noteprob', help='Select the note by taking a random note with probabilities based on \
        the neural network''s output. Default behavior', action='store_true')
group.add_argument('--noteeprob', metavar=('epsilon'), help='Behaves like --noteprob with probability epsilon, and --notemax otherwise.', type=float)
group.add_argument('--noteedprob', metavar=('gamma'), help='Behaves like --noteprob with probability e, where e starts at one, and is \
        multiplied by gamma every note (exponential decay). Behaves like --notemax the rest of the time.', type=float)
args = parser.parse_args()

data  = np.load(args.data)
msgs  = data['messages']
inds  = data['indices']
names = data['names']
maxes = data['maxes']

# make directories for saving songs
songdir = os.path.dirname(args.songprefix)
if not os.path.exists(songdir):
    os.makedirs(songdir)

# select note selection function
# note_fun takes a list of arrays of probabilities
def note_f_max(prob):
    return np.argmax(prob)
def note_f_prob(prob):
    return np.random.choice(len(prob), p=prob)
if args.notemax:
    print('Note selection set to argmax')
    note_f = lambda prob, i: note_f_max(prob)
elif args.noteedprob is not None:
    print('Note selection set to exponential decay probability. Gamma = ' + str(args.noteedprob))
    note_f = lambda prob, i: note_f_max(prob) if random.random() >= (args.noteedprob ** i) else note_f_prob(prob)
elif args.noteeprob is not None:
    print('Note selection set to epsilon probability. Epsilon = ' + str(args.noteeprob))
    note_f = lambda prob, i: note_f_max(prob) if random.random() >= args.noteeprob else note_f_prob(prob)
else:
    print('Note selection set to softmax')
    note_f = lambda prob, i: note_f_prob(prob)


def get_batch(msgs, inds, time_steps, batch_size):
    batch = np.zeros((time_steps, batch_size))
    for b in range(batch_size):
        start = random.randint(0, inds.shape[0] - time_steps)
        batch[:,b] = inds[start:start+time_steps]
    return batch

with tf.Session() as sess:
    ################################################################################
    ##                           GRAPH DEFINITION                                 ##
    ################################################################################

    params = {}
    params['RNN_SIZES']     = args.rnns
    params['DENSE_SIZES']   = args.denses
    params['CATEGORIES']    = msgs.shape[0]
    params['LEARNING_RATE'] = 1e-2
    params['SAVE_DIR']      = args.savedir
    
    # print information about the neural network
    print('Neural network parameters')
    for k,v in params.items():
        print(str(k) + ': ' + str(v))
    net = model.Net(sess, params)

    # attempt to restore
    try:
        net.restore()
    except:
        sess.run(tf.global_variables_initializer())


    ################################################################################
    ##                           TRAINING LOOP                                    ##
    ################################################################################

    TIME_STEPS = 50
    LOSS_TIME_STEPS = 49
    NUM_EPOCHS = args.epochs
    TRAIN_STEPS = args.train
    BATCH_SIZE = 100
    SONG_LENGTH = args.songlength

    for epoch in range(NUM_EPOCHS):
        for ti in range(TRAIN_STEPS):
            batch = get_batch(msgs, inds, time_steps=TIME_STEPS, batch_size=BATCH_SIZE)
            net.train(batch, LOSS_TIME_STEPS)

        if TRAIN_STEPS > 0:
            # add summary of performance
            batch = get_batch(msgs, inds, time_steps=TIME_STEPS, batch_size=BATCH_SIZE)
            summaries = net.summarize(batch, LOSS_TIME_STEPS)

            # save net
            net.save()
            print('Saved snapshot of model at iteration ' + str(net.global_step()))

        if (epoch+1) % args.generate == 0:
            # make a song of length to test
            indices = np.zeros((SONG_LENGTH,), dtype=np.int32)
            in_state = None
            in_inds = get_batch(msgs, inds, time_steps=TIME_STEPS, batch_size=1)
            #in_msg = np.array([random.randint(0,m-1) for m in maxes], ndmin=2)
            #in_msg = np.array(np.concatenate(([0,5], maxes[2:])), ndmin=2) # middle C

            for i in range(SONG_LENGTH):
                # use network to get probabilities of each piece of message
                out_prob, out_state = net.predict(in_inds, in_state)

                # randomly select based on output values, which should sum to one
                out_prob_squeezed = np.squeeze(out_prob, axis=(0,1))
                out_ind = np.array(note_f(out_prob_squeezed, i), ndmin=2)

                # append to song
                indices[i] = out_ind

                # print out as we generate the song
                #print("IN: " + str(in_msg))
                #print("OUT: " + str(out_msg))

                # take output as next input
                in_state = out_state
                in_inds = out_ind

            # save the song
            midifile = postprocess(msgs, indices, volumes=maxes[2], duration_categories=maxes[3])
            songfilename = args.songprefix + str(net.global_step()) + '-' + str(datetime.now()).replace(' ','_').replace(':','-').replace('.','-') + '.mid'
            midifile.save(songfilename)
            print('Saved a new song at ' + songfilename)
