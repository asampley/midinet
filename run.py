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
maxes = data['maxes']
names = data['names']

# make directories for saving songs
songdir = os.path.dirname(args.songprefix)
if not os.path.exists(songdir):
    os.makedirs(songdir)

# select note selection function
# note_fun takes a list of arrays of probabilities
def note_f_max(probs):
    return [np.argmax(prob) for prob in probs]
def note_f_prob(probs):
    return [np.random.choice(len(prob), p=prob) for prob in probs]
if args.notemax:
    print('Note selection set to argmax')
    note_f = lambda probs, i: note_f_max(probs)
elif args.noteedprob is not None:
    print('Note selection set to exponential decay probability. Gamma = ' + str(args.noteedprob))
    note_f = lambda probs, i: note_f_max(probs) if random.random() >= (args.noteedprob ** i) else note_f_prob(probs)
elif args.noteeprob is not None:
    print('Note selection set to epsilon probability. Epsilon = ' + str(args.noteeprob))
    note_f = lambda probs, i: note_f_max(probs) if random.random() >= args.noteeprob else note_f_prob(probs)
else:
    print('Note selection set to softmax')
    note_f = lambda probs, i: note_f_prob(probs)


def get_batch(data, time_steps, batch_size):
    batch = np.zeros((time_steps, batch_size, data.shape[1])) 
    for b in range(batch_size):
        start = random.randint(0, data.shape[0] - time_steps)
        batch[:,b,:] = data[start:start+time_steps,:]
    return batch

with tf.Session() as sess:
    ################################################################################
    ##                           GRAPH DEFINITION                                 ##
    ################################################################################

    params = {}
    params['RNN_SIZES']     = [512, 512]
    params['DATA_SIZES']    = maxes
    params['DATA_NAMES']    = names
    params['DATA_WEIGHTS']  = [1.0, 1.0, 1.0, 1.0]
    params['LEARNING_RATE'] = 1e-4
    params['SAVE_DIR']      = args.savedir

    net = model.Net(sess, params)

    # attempt to restore
    try:
        net.restore()
    except:
        sess.run(tf.global_variables_initializer())


    ################################################################################
    ##                           TRAINING LOOP                                    ##
    ################################################################################

    TIME_STEPS = 101
    LOSS_TIME_STEPS = 100
    NUM_EPOCHS = args.epochs
    TRAIN_STEPS = args.train
    BATCH_SIZE = 1000
    SONG_LENGTH = args.songlength

    for epoch in range(NUM_EPOCHS):
        for ti in range(TRAIN_STEPS):
            batch = get_batch(msgs, time_steps=TIME_STEPS, batch_size=BATCH_SIZE)
            net.train(batch, LOSS_TIME_STEPS)

        if TRAIN_STEPS > 0:
            # add summary of performance
            batch = get_batch(msgs, time_steps=TIME_STEPS, batch_size=BATCH_SIZE)
            summaries = net.summarize(batch, LOSS_TIME_STEPS)

            # save net
            net.save()
            print('Saved snapshot of model at iteration ' + str(net.global_step()))

        if (epoch+1) % args.generate == 0:
            # make a song of length to test
            messages = np.zeros((SONG_LENGTH, msgs.shape[1]), dtype=np.int32)
            in_state = None
            in_msg = np.array([random.randint(0,m-1) for m in maxes], ndmin=3)
            #in_msg = np.array(np.concatenate(([0,5], maxes[2:])), ndmin=3) # middle C

            for i in range(SONG_LENGTH):
                # use network to get probabilities of each piece of message
                out_probs, out_state = net.predict(in_msg, in_state)

                # randomly select based on output values, which should sum to one
                out_probs_squeezed = [np.squeeze(out_prob, axis=(0,1)) for out_prob in out_probs]
                out_msg = np.array(note_f(out_probs_squeezed, i), ndmin=3)

                # append to song
                messages[i,:] = out_msg

                # print out as we generate the song
                #print("IN: " + str(in_msg))
                #print("OUT: " + str(out_msg))

                # take output as next input
                in_state = out_state
                in_msg = out_msg

            # save the song
            midifile = postprocess(messages)
            songfilename = args.songprefix + str(net.global_step()) + '-' + str(datetime.now()).replace(' ','_').replace(':','-').replace('.','-') + '.mid'
            midifile.save(songfilename)
            print('Saved a new song at ' + songfilename)
