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
    start = np.random.randint(0, data.shape[0]) # we'll miss a bit of data at the end, but that's okay

    instance = np.zeros((time_steps, data.shape[1] - 1))
    next_note = np.zeros((1, data.shape[1] - 1))
    i = 0
    i_instance = 0
    while i_instance < instance.shape[0] and start + i < data.shape[0]:
        repeats = int(min(data[start+i,0], time_steps - i_instance))
        notes = data[start+i,1:].reshape((1,128))
        instance[i_instance:i_instance+repeats,:] = np.repeat(notes, repeats, axis=0)
        i_instance += repeats
        i += 1

    if start + i < data.shape[0]:
        next_note = data[start+i,1:].reshape((1,128))

#    np.set_printoptions(threshold=np.inf)
#    for i in range(instance.shape[0]):
#        print(np.where(instance[i,:] != 0)[0])

    return instance, next_note

def generate_batch(data, time_steps, batch_size):
    """Generates instance of a problem.
    Returns
    -------
    x: np.array
        two numbers to be added represented by bits.
        shape: b, i, n
        where:
            b is bit index from the end
            i is example idx in batch
            n is one of [0,1] depending for first and
                second summand respectively
    y: np.array
        the result of the addition
        shape: b, i, n
        where:
            b is bit index from the end
            i is example idx in batch
            n is always 0
    """
    num_notes = data.shape[1] - 1
    x = np.empty((time_steps, batch_size, num_notes))
    y = np.empty((1,          batch_size, num_notes))

    for i in range(batch_size):
        notes, next_note = get_instance(data, time_steps)
        x[:, i, :] = notes
        y[:, i, :] = next_note
    return x, y


with tf.Session() as sess:
    ################################################################################
    ##                           GRAPH DEFINITION                                 ##
    ################################################################################

    params = {}
    params['NUM_NOTES']     = 128
    params['RNN_HIDDEN']    = 512
    params['LEARNING_RATE'] = 1e-3
    params['NUM_LAYERS']    = 4

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
            x, y = generate_batch(data=data, time_steps=TIME_STEPS, batch_size=BATCH_SIZE)
            net.train(x, y)

            if ti % 10 == 0:
                summaries = net.summarize(x, y)

        # save net
        net.save()
        print('Saved snapshot of model')

        # make a song of length to test
        next_state = None
        next_input = np.random.randint(2, size=(1, 1, 128)).astype(np.float32)
        song = np.zeros((SONG_LENGTH, 129), dtype=np.float32)

        for i in range(SONG_LENGTH):
            #predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            #    x={"x": next_input},
            #    num_epochs=1,
            #    shuffle=False)
            #next_input_gen = estimator.predict(predict_input_fn)
            #output = next(next_input_gen)
            
            output, next_state = net.predict(next_input, next_state)

            np.set_printoptions(threshold=np.inf)
            #print("IN:  " + str(next_input))
            #print("OUT: " + str(output))
            print("IN:  " + str(np.where(next_input >= 0.5)[2]))
            print("OUT: " + str(np.where(output >= 0.5)[1]))

            song[i,0] = 1
            song[i,1:] = output
            next_input = np.reshape(output, (1,1,128))

        # clamp song to 0 or 1 for volume of note
        song[song < 0.5] = 0
        song[song >= 0.5] = 1
        
        # save the song
        midifile = postprocess(song)
        songfilename = 'songs/%s.mid'%(net.global_step())
        midifile.save(songfilename)
        print('Saved a new song at ' + songfilename)
