# Credit to https://gist.github.com/siemanko/b18ce332bde37e156034e5d3f60f8a23 for initial code

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import model
from preprocess import postprocess

################################################################################
##                           DATASET GENERATION                               ##
##                                                                            ##
##  The problem we are trying to solve is adding two binary numbers. The      ##
##  numbers are reversed, so that the state of RNN can add the numbers        ##
##  perfectly provided it can learn to store carry in the state. Timestep t   ##
##  corresponds to bit len(number) - t.                                       ##
################################################################################

data = np.load('data/all.npy')

def get_instance(data, time_steps):
    start = np.random.randint(0, data.shape[0] - time_steps) # we'll miss a bit of data at the end, but that's okay

    notes = np.zeros((time_steps, data.shape[1] - 1))
    durations = np.zeros((time_steps, 8)) # (1/32 note, 1/16 note, ..., 4 whole notes)

    i = 0
    i_data = 0
    remaining_duration = 0
    while i < time_steps:
        remaining_duration = int(data[start+i_data, 0])
        
        # reduce durations into intervals of at most 4 whole notes
        i_duration = 7
        while remaining_duration > 0 and i < time_steps:
            if remaining_duration >= 2 ** i_duration:
                notes[i,:] = data[start+i_data, 1:].reshape((1,128))
                durations[i,i_duration] = 1
                remaining_duration = remaining_duration - 2 ** i_duration
            else:
                i_duration = i_duration - 1
            i = i + 1
        i_data = i_data + 1
            
#    np.set_printoptions(threshold=np.inf)
#    for i in range(notes.shape[0]):
#        print(np.where(notes[i,:] != 0)[0])

    return notes, durations

def generate_batch(data, time_steps, batch_size):
    num_notes = data.shape[1] - 1
    notes     = np.empty((time_steps, batch_size, num_notes))
    durations = np.empty((time_steps, batch_size, 8        ))

    for i in range(batch_size):
        notes_i, durations_i = get_instance(data, time_steps)
        notes[:, i, :] = notes_i
        durations[:, i, :] = durations_i
    return notes, durations


with tf.Session() as sess:
    ################################################################################
    ##                           GRAPH DEFINITION                                 ##
    ################################################################################

    params = {}
    params['NUM_NOTES']     = 128
    params['RNN_HIDDEN']    = 256
    params['LEARNING_RATE'] = 1e-4
    params['NUM_LAYERS']    = 2

    #estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir='model', params=params)
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
    BATCH_SIZE = 200
    SONG_LENGTH = 640

    def input_fn():
        x, y = generate_batch(data=data, time_steps=TIME_STEPS, batch_size=BATCH_SIZE)
        return { 'x': tf.constant(x, dtype=tf.float32) }, tf.constant(y, dtype=tf.float32)

    for epoch in range(NUM_EPOCHS):
        #estimator.train(input_fn=input_fn, steps=TRAIN_STEPS)
        #eval_metrics = estimator.evaluate(input_fn=input_fn, steps=10)
        #print("Epoch %d: %s"%(epoch, eval_metrics))
        for ti in range(TRAIN_STEPS):
            notes, durations = generate_batch(data=data, time_steps=TIME_STEPS, batch_size=BATCH_SIZE)
            net.train(notes, durations, LOSS_TIME_STEPS)

            if ti % 10 == 0:
                summaries = net.summarize(notes, durations, LOSS_TIME_STEPS)

        # save net
        net.save()
        print('Saved snapshot of model')

        # make a song of length to test
        next_state = None
        next_notes = np.zeros((1, 1, 128), dtype=np.float32)
        next_notes[0,0,81] = 1
        next_durations = np.zeros((1, 1, 8), dtype=np.float32)
        next_durations[0,0,6] = 1
        song = np.zeros((SONG_LENGTH, 129), dtype=np.float32)

        for i in range(SONG_LENGTH):
            #predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            #    x={"x": next_input},
            #    num_epochs=1,
            #    shuffle=False)
            #next_input_gen = estimator.predict(predict_input_fn)
            #output = next(next_input_gen)
            
            note, duration, next_state = net.predict(next_notes, next_durations, next_state)

            np.set_printoptions(threshold=np.inf)
            #print("IN:  " + str(next_input))
            #print("OUT: " + str(output))
            print("IN:  " + str(np.where(next_notes >= 0.5)[2]))
            print("OUT: " + str(np.where(note >= 0.5)[2]))

            song[i,0] = 2 ** np.argmax(duration)
            song[i,1:] = note
            next_notes = np.reshape(note, (1,1,128))
            next_durations = np.reshape(duration, (1,1,8))

        # clamp song to 0 or 1 for volume of note
        song[song < 0.5] = 0
        song[song >= 0.5] = 1
        
        # save the song
        midifile = postprocess(song)
        songfilename = 'songs/%s.mid'%(net.global_step())
        midifile.save(songfilename)
        print('Saved a new song at ' + songfilename)
