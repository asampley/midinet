import tensorflow as tf

class Net:
    def __init__(self, session, params):
        self.session = session

        NUM_NOTES     = params['NUM_NOTES']
        RNN_HIDDEN    = params['RNN_HIDDEN']
        LEARNING_RATE = params['LEARNING_RATE']
        NUM_LAYERS    = params['NUM_LAYERS']

        self.notes = tf.placeholder(tf.float32, (None, None, NUM_NOTES), 'notes')  # (time, batch, notes)
        self.durations = tf.placeholder(tf.float32, (None, None, 8), 'durations')  # (time, batch, duration_categories)

        self.notes_in = tf.cond(tf.shape(self.notes)[0] > 1, lambda: self.notes[:-1,:,:], lambda: self.notes)
        self.durations_in = tf.cond(tf.shape(self.durations)[0] > 1, lambda: self.durations[:-1,:,:], lambda: self.durations)
            # don't use last piece of data, to have the labels offset by one from input
            # if only one time step, it's prediction, and we shouldn't reduce input
            # TODO: make this more elegant
        self.loss_time_steps = tf.placeholder(tf.int32, name='loss_time_steps')

        self.notes_labels = self.notes_in[-self.loss_time_steps:, :, :]
        self.durations_labels = self.durations_in[-self.loss_time_steps:, :, :]

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
        self.batch_size    = tf.shape(self.notes_in)[1]
        with tf.name_scope("State"):
            self.states = Net.get_state_variables(self.batch_size, self.cell)
            #self.states = self.cell.zero_state(self.batch_size, tf.float32)

        #print(self.states)

        # Given inputs (time, batch, num_notes) outputs a tuple
        #  - outputs: (time, batch, num_notes)
        #  - states:  (time, batch, hidden_size)
        rnn_outputs, self.state_out = tf.nn.dynamic_rnn(self.cell, self.notes_in, initial_state=self.states, time_major=True)

        rnn_outputs_last_times = rnn_outputs[-self.loss_time_steps:,:,:]
        initializer = tf.contrib.layers.xavier_initializer()
        fc1_size = 4 * NUM_NOTES
        
        # fully connected layer from rnn to note output
        with tf.name_scope("FC_Notes_1"):
            W1 = tf.Variable(initializer((RNN_HIDDEN, fc1_size)), name='W')
            b1 = tf.Variable(initializer((fc1_size,)), name='b')
            self.fc1 = tf.tensordot(rnn_outputs_last_times, W1, [[2],[0]]) + b1
            self.fc1 = tf.nn.leaky_relu(self.fc1, 0.2)
            self.fc1 = tf.layers.dropout(self.fc1, rate=self.keep_prob)

        with tf.name_scope("FC_Notes_2"):
            W2 = tf.Variable(initializer((fc1_size, NUM_NOTES)), name='W')
            b2 = tf.Variable(initializer((NUM_NOTES,)), name='b')
            self.fc2 = tf.tensordot(self.fc1, W2, [[2],[0]]) + b2
            self.fc2 = tf.nn.leaky_relu(self.fc2, 0.2)
        
        # fully connected layer from rnn to duration output
        with tf.name_scope("FC_Duration_1"):
            W1 = tf.Variable(initializer((RNN_HIDDEN, 8)), name='W')
            b1 = tf.Variable(initializer((8,)), name='b')
            self.fcd1 = tf.tensordot(rnn_outputs_last_times, W1, [[2],[0]]) + b1
            self.fcd1 = tf.nn.softmax(self.fcd1, name='Softmax')

        # outputs
        self.note_out = self.fc2
        self.duration_out = self.fcd1

        # compute elementwise L2 norm
        with tf.name_scope("Error"):
            error = tf.reduce_mean(tf.square(self.notes_labels - self.note_out)) + tf.reduce_mean(tf.square(self.durations_labels - self.duration_out))

        # optimize
        self.train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error, global_step=self._global_step)

        # assuming that absolute difference between output and correct answer is 0.5
        # or less we can round it to the correct output.
        with tf.name_scope("Note_Accuracy"):
            accuracy = 1 - tf.reduce_mean(tf.abs(self.notes_labels - tf.cast(self.note_out > 0.5, tf.float32)))
        with tf.name_scope("Duration_Accuracy"):
            duration_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.duration_out, axis=2),tf.argmax(self.durations_labels, axis=2)), tf.float32))
        
        # Make summary op and file
        tf.summary.scalar('note_accuracy', accuracy)
        tf.summary.scalar('duration_accuracy', duration_accuracy)
        tf.summary.scalar('error', error)
        tf.summary.histogram('note_outputs', self.note_out)
        tf.summary.histogram('note_outputs_rounded', tf.round(self.note_out))
        tf.summary.histogram('notes_labels', self.notes_labels)
        tf.summary.histogram('duration_outputs', tf.argmax(self.duration_out, axis=2))
        tf.summary.histogram('duration_labels', tf.argmax(self.durations_labels, axis=2))

        self.summaries = tf.summary.merge_all()
        self.summaryFileWriter = tf.summary.FileWriter('model', self.session.graph)

        # Make net saver
        self.saver = tf.train.Saver()

    def save(self):
        self.saver.save(self.session, 'model/model.ckpt')

    def restore(self):
        self.saver.restore(self.session, 'model/model.ckpt')
    
    def train(self, note_input, dur_input, loss_time_steps, batch_state = None, keep_prob = 0.5):
        feed_dict = {
            self.notes: note_input,
            self.durations: dur_input,
            self.loss_time_steps: loss_time_steps,
            self.keep_prob: keep_prob
        }
        if batch_state is not None:
            feed_dict[self.states] = batch_state

        return self.session.run(
            [self.train_fn, self.state_out],
            feed_dict=feed_dict)

    def predict(self, note_input, dur_input, batch_state = None):
        """
        Returns the outputs and new state of the lstm
        """
        
        feed_dict = {
            self.notes: note_input,
            self.durations: dur_input,
            self.loss_time_steps: 2,
            self.keep_prob: 1.0
        }
        if batch_state is not None:
            feed_dict[self.states] = batch_state

        return self.session.run(
            [self.note_out, self.duration_out, self.state_out],
            feed_dict=feed_dict)

    def summarize(self, note_input, dur_input, loss_time_steps, batch_state = None):
        feed_dict = {
            self.notes: note_input,
            self.durations: dur_input,
            self.loss_time_steps: loss_time_steps,
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
            state_init_c = tf.tile(tf.Variable(tf.zeros((1, state_c.shape[1]))), (batch_size, 1))
            state_init_h = tf.tile(tf.Variable(tf.zeros((1, state_h.shape[1]))), (batch_size, 1))
            state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                tf.placeholder_with_default(state_init_c, state_c.shape, "State_C"),
                tf.placeholder_with_default(state_init_h, state_h.shape, "State_H")))
                #tf.Variable(state_c, trainable=False),
                #tf.Variable(state_h, trainable=False)))
        # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
        return tuple(state_variables)
