import tensorflow as tf

class Net:
    def __init__(self, session, params):
        self.session = session

        NUM_NOTES     = params['NUM_NOTES']
        RNN_HIDDEN    = params['RNN_HIDDEN']
        LEARNING_RATE = params['LEARNING_RATE']
        NUM_LAYERS    = params['NUM_LAYERS']

        self.inputs = tf.placeholder(tf.float32, (None, None, NUM_NOTES), 'Input')  # (time, batch, notes)
        self.labels = tf.placeholder(tf.float32, (None, None, NUM_NOTES), 'Label') # (time, batch, notes)
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

        # Set variable for dropout of each layer
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

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
            cells[i] = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True, activation=tf.nn.relu)
            cells[i] = tf.nn.rnn_cell.DropoutWrapper(cells[i], output_keep_prob=self.keep_prob)
        self.cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        # Create initial state. Here it is just a constant tensor filled with zeros,
        # but in principle it could be a learnable parameter. This is a bit tricky
        # to do for LSTM's tuple state, but can be achieved by creating two vector
        # Variables, which are then tiled along batch dimension and grouped into tuple.
        self.batch_size    = tf.shape(self.inputs)[1]
        with tf.name_scope("State"):
            self.states = Net.get_state_variables(self.batch_size, self.cell)
            #self.states = self.cell.zero_state(self.batch_size, tf.float32)

        #print(self.states)

        # Given inputs (time, batch, num_notes) outputs a tuple
        #  - outputs: (time, batch, num_notes)
        #  - states:  (time, batch, hidden_size)
        rnn_outputs, self.state_out = tf.nn.dynamic_rnn(self.cell, self.inputs, initial_state=self.states, time_major=True)

        # fully connected layer from rnn to last output
        rnn_outputs_last_time = rnn_outputs[-1,:,:]
        with tf.name_scope("FC1"):
            self.fc1 = tf.layers.dense(rnn_outputs_last_time, 4 * NUM_NOTES)
            self.fc1 = tf.nn.leaky_relu(self.fc1, 0.2)
            self.fc1 = tf.layers.dropout(self.fc1, rate=self.keep_prob)

        with tf.name_scope("FC2"):
            self.fc2 = tf.layers.dense(self.fc1, NUM_NOTES)
            self.fc2 = tf.nn.leaky_relu(self.fc2, 0.2)
            self.fc2 = tf.layers.dropout(self.fc2, rate=self.keep_prob)

        # output layer
        self.outputs = self.fc2

        # compute elementwise L2 norm
        with tf.name_scope("Error"):
            error = tf.reduce_mean(tf.square(self.labels - self.outputs))

        # optimize
        self.train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error, global_step=self._global_step)

        # assuming that absolute difference between output and correct answer is 0.5
        # or less we can round it to the correct output.
        with tf.name_scope("Accuracy"):
            accuracy = 1 - tf.reduce_mean(tf.abs(self.labels - tf.cast(self.outputs > 0.5, tf.float32)))
        
        # Make summary op and file
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('error', error)
        tf.summary.histogram('fc2', self.fc2)
        tf.summary.histogram('outputs', self.outputs)
        tf.summary.histogram('outputs rounded', tf.round(self.outputs))
        tf.summary.histogram('labels', self.labels)

        self.summaries = tf.summary.merge_all()
        self.summaryFileWriter = tf.summary.FileWriter('model', self.session.graph)

        # Make net saver
        self.saver = tf.train.Saver()

    def save(self):
        self.saver.save(self.session, 'model/model.ckpt')

    def restore(self):
        self.saver.restore(self.session, 'model/model.ckpt')
    
    def train(self, batch_input, batch_labels, batch_state = None, keep_prob = 0.5):
        feed_dict = {
            self.inputs: batch_input,
            self.labels: batch_labels,
            self.keep_prob: keep_prob
        }
        if batch_state is not None:
            feed_dict[self.states] = batch_state

        return self.session.run(
            [self.train_fn, self.state_out],
            feed_dict=feed_dict)

    def predict(self, batch_input, batch_state = None):
        """
        Returns the outputs and new state of the lstm
        """
        
        feed_dict = {
            self.inputs: batch_input,
            self.keep_prob: 1.0
        }
        if batch_state is not None:
            feed_dict[self.states] = batch_state

        return self.session.run(
            [self.outputs, self.state_out],
            feed_dict=feed_dict)

    def summarize(self, batch_input, batch_labels, batch_state = None):
        feed_dict = {
            self.inputs: batch_input,
            self.labels: batch_labels,
            self.keep_prob: 1.0
        }
        if batch_state is not None:
            feed_dict[self.states] = batch_state

        summaries, step = self.session.run(
            [self.summaries, self._global_step],
            feed_dict = feed_dict)
        self.summaryFileWriter.add_summary(summaries, step)
        self.summaryFileWriter.flush()

    def global_step(self):
        return self.session.run([self._global_step])[0]

    @staticmethod
    def get_state_variables(batch_size, cell):
        # For each layer, get the initial state and make a variable out of it
        # to enable updating its value.
        state_variables = []
        for state_c, state_h in cell.zero_state(batch_size, tf.float32):
            state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                tf.placeholder_with_default(state_c, state_c.shape, "State_C"),
                tf.placeholder_with_default(state_h, state_h.shape, "State_H")))
                #tf.Variable(state_c, trainable=False),
                #tf.Variable(state_h, trainable=False)))
        # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
        return tuple(state_variables)
