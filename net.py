# Credit to https://gist.github.com/siemanko/b18ce332bde37e156034e5d3f60f8a23 for initial code

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import model
import random
from preprocess import postprocess
import os

# make directories for saving things
if not os.path.exists('songs'):
    os.makedirs('songs')

data  = np.load('data/all.npz')
msgs  = data['messages']
maxes = data['maxes']
names = data['names']

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
    params['RNN_SIZES']     = [64, 64]
    params['DATA_SIZES']    = maxes
    params['DATA_NAMES']    = names
    params['LEARNING_RATE'] = 1e-4

    net = model.Net(sess, params)

    # attempt to restore
    try:
        net.restore()
    except:
        sess.run(tf.global_variables_initializer())


    ################################################################################
    ##                           TRAINING LOOP                                    ##
    ################################################################################

    TIME_STEPS = 100
    LOSS_TIME_STEPS = 50
    NUM_EPOCHS = 1000
    TRAIN_STEPS = 100
    BATCH_SIZE = 50
    SONG_LENGTH = 640

    for epoch in range(NUM_EPOCHS):
        for ti in range(TRAIN_STEPS):
            batch = get_batch(msgs, time_steps=TIME_STEPS, batch_size=BATCH_SIZE)
            net.train(batch, LOSS_TIME_STEPS)

            if ti % 10 == 0:
                summaries = net.summarize(batch, LOSS_TIME_STEPS)

        # save net
        net.save()
        print('Saved snapshot of model')

        # make a song of length to test
        messages = np.zeros((SONG_LENGTH, msgs.shape[1]), dtype=np.int32)
        in_state = None
        in_msg = np.array([random.randint(0,m-1) for m in maxes], ndmin=3)

        for i in range(SONG_LENGTH):            
            # use network to get probabilities of each piece of message
            out_probs, out_state = net.predict(in_msg, in_state)

            # randomly select based on output values, which should sum to one
            out_probs_squeezed = map(np.squeeze, out_probs)
            out_msg = np.array([np.random.choice(len(prob), p=prob) for prob in out_probs_squeezed], ndmin=3)

            # append to song
            messages[i,:] = out_msg

            # print out as we generate the song
            print("IN: " + str(in_msg))
            print("OUT: " + str(out_msg))

            # take output as next input
            in_state = out_state
            in_msg = out_msg

        # save the song
        midifile = postprocess(messages)
        songfilename = 'songs/%s.mid'%(net.global_step())
        midifile.save(songfilename)
        print('Saved a new song at ' + songfilename)
