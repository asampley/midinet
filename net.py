# Credit to https://gist.github.com/siemanko/b18ce332bde37e156034e5d3f60f8a23 for initial code

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import model
from preprocess import postprocess

data = np.load('data/all.npz')
reps = data['reps']
notes = data['notes']

def get_instance(reps, notes, time_steps):
    start = np.random.randint(0, reps.shape[0] - time_steps) # we'll miss a bit of data at the end, but that's okay

    notes_instance = np.zeros((time_steps, notes.shape[1], notes.shape[2]))
    durations = np.zeros((time_steps, 8)) # (1/32 note, 1/16 note, ..., 4 whole notes)

    i = 0
    i_data = 0
    remaining_duration = 0
    while i < time_steps:
        remaining_duration = int(reps[start+i_data])
        
        # reduce durations into intervals of at most 4 whole notes
        i_duration = 7
        while remaining_duration > 0 and i < time_steps:
            if remaining_duration >= 2 ** i_duration:
                notes_instance[i,:,:] = notes[start+i_data,:,:]
                durations[i,i_duration] = 1
                remaining_duration = remaining_duration - 2 ** i_duration
            else:
                i_duration = i_duration - 1
            i = i + 1
        i_data = i_data + 1
            
#    np.set_printoptions(threshold=np.inf)
#    for i in range(notes.shape[0]):
#        print(np.where(notes[i,:] != 0)[0])

    return notes_instance, durations

def generate_batch(reps, notes, time_steps, batch_size):
    notes_batch = np.empty((time_steps, batch_size, notes.shape[1], notes.shape[2]))
    durations   = np.empty((time_steps, batch_size, 8        ))

    for i in range(batch_size):
        notes_i, durations_i = get_instance(reps, notes, time_steps)
        notes_batch[:, i, :, :] = notes_i
        durations[:, i, :] = durations_i
    return notes_batch, durations


with tf.Session() as sess:
    ################################################################################
    ##                           GRAPH DEFINITION                                 ##
    ################################################################################

    params = {}
    params['RNN_NOTES_CHANNELS']    = [64, 64]
    params['RNN_DURATION_CHANNELS'] = [16, 16]
    params['LEARNING_RATE']         = 1e-4

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
    BATCH_SIZE = 50
    SONG_LENGTH = 640

    def input_fn():
        x, y = generate_batch(notes, reps, time_steps=TIME_STEPS, batch_size=BATCH_SIZE)
        return { 'x': tf.constant(x, dtype=tf.float32) }, tf.constant(y, dtype=tf.float32)

    for epoch in range(NUM_EPOCHS):
        #estimator.train(input_fn=input_fn, steps=TRAIN_STEPS)
        #eval_metrics = estimator.evaluate(input_fn=input_fn, steps=10)
        #print("Epoch %d: %s"%(epoch, eval_metrics))
        for ti in range(TRAIN_STEPS):
            notes_batch, durations_batch = generate_batch(reps, notes, time_steps=TIME_STEPS, batch_size=BATCH_SIZE)
            net.train(notes_batch, durations_batch, LOSS_TIME_STEPS)

            if ti % 10 == 0:
                summaries = net.summarize(notes_batch, durations_batch, LOSS_TIME_STEPS)

        # save net
        net.save()
        print('Saved snapshot of model')

        # make a song of length to test
        next_state = None
        next_notes = np.random.randint(0, 2, size=(1, 1, 12, 10)).astype(np.float32)
        next_durations = np.zeros((1, 1, 8), dtype=np.float32)
        next_durations[0,0,6] = 1
        song_reps = np.zeros((SONG_LENGTH,), dtype=np.float32)
        song = np.zeros((SONG_LENGTH, 12, 10), dtype=np.float32)

        for i in range(SONG_LENGTH):
            #predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            #    x={"x": next_input},
            #    num_epochs=1,
            #    shuffle=False)
            #next_input_gen = estimator.predict(predict_input_fn)
            #output = next(next_input_gen)
            
            print("IN:  " + str(2 ** np.argmax(next_durations)) + ":" + str(12 * np.where(next_notes >= 0.5)[3] + np.where(next_notes >= 0.5)[2]))

            note_out, duration_out, next_state = net.predict(next_notes, next_durations, next_state)

            print(duration_out[0,0,:])
            duration_i = np.random.choice(8, p=duration_out[0,0,:])
            song_reps[i] = 2 ** duration_i
            # randomly select notes based on outputs as probability
            song[i,:,:] = np.random.random(note_out.shape) < note_out

            print("OUT: " + str(song_reps[i]) + ":" + str(12 * np.where(song[i,:,:] >= 0.5)[1] + np.where(song[i,:,:] >= 0.5)[0]))

            next_notes = np.reshape(song[i,:,:], (1,1,12,10))
            next_durations = np.zeros((1,1,8), dtype=np.float32)
            next_durations[0,0,duration_i] = 1
        
        # save the song
        midifile = postprocess(song_reps, song)
        songfilename = 'songs/%s.mid'%(net.global_step())
        midifile.save(songfilename)
        print('Saved a new song at ' + songfilename)
