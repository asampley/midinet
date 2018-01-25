import tensorflow as tf

def model_fn(features, labels, mode, params):
    NUM_NOTES     = params['NUM_NOTES']
    RNN_HIDDEN    = params['RNN_HIDDEN']
    LEARNING_RATE = params['LEARNING_RATE']
    NUM_LAYERS    = params['NUM_LAYERS']

    #inputs  = tf.placeholder(tf.float32, (None, None, NUM_NOTES))  # (time, batch, notes)
    inputs = features['x']
    #outputs = tf.placeholder(tf.float32, (None, None, NUM_NOTES)) # (time, batch, notes)

    ## Here cell can be any function you want, provided it has two attributes:
    #     - cell.zero_state(batch_size, dtype)- tensor which is an initial value
    #                                           for state in __call__
    #     - cell.__call__(input, state) - function that given input and previous
    #                                     state returns tuple (output, state) where
    #                                     state is the state passed to the next
    #                                     timestep and output is the tensor used
    #                                     for infering the output at timestep. For
    #                                     example for LSTM, output is just hidden,
    #                                     but state is memory + hidden
    # Example LSTM cell with learnable zero_state can be found here:
    #    https://gist.github.com/nivwusquorum/160d5cf7e1e82c21fad3ebf04f039317
    cells = [None for _ in range(NUM_LAYERS)]
    for i in range(NUM_LAYERS):
        cells[i] = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
        cells[i] = tf.nn.rnn_cell.DropoutWrapper(cells[i], output_keep_prob=0.5)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

    # Create initial state. Here it is just a constant tensor filled with zeros,
    # but in principle it could be a learnable parameter. This is a bit tricky
    # to do for LSTM's tuple state, but can be achieved by creating two vector
    # Variables, which are then tiled along batch dimension and grouped into tuple.
    batch_size    = tf.shape(inputs)[1]
    initial_state = cell.zero_state(batch_size, tf.float32)

    # Given inputs (time, batch, num_notes, input_size) outputs a tuple
    #  - outputs: (time, batch, num_notes, output_size)  [do not mistake with OUTPUT_SIZE]
    #  - states:  (time, batch, hidden_size)
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)

    # project output from rnn output size to NUM_NOTES. Sometimes it is worth adding
    # an extra layer here.
    final_projection = lambda x: tf.contrib.layers.fully_connected(x, NUM_NOTES)

    # apply projection to every timestep.
    predicted_outputs = tf.map_fn(final_projection, rnn_outputs)

    if mode != tf.estimator.ModeKeys.PREDICT:
        # compute elementwise L2 norm
        error = tf.reduce_mean(tf.square(labels - predicted_outputs))

        # optimize
        train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error, global_step=tf.train.get_global_step())

        # assuming that absolute difference between output and correct answer is 0.5
        # or less we can round it to the correct output.
        accuracy = tf.reduce_mean(tf.cast(tf.abs(labels - predicted_outputs) < 0.5, tf.float32))
        
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('error', error)

        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predicted_outputs,
                loss=error,
                train_op=train_fn)
    else:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predicted_outputs)
