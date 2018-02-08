import tensorflow as tf

class Net:
    def __init__(self, session, params):
        self.session = session

        PITCHES = 12  # pitches in an octave
        OCTAVES = 11  # octaves in a midi file
        DURATIONS = 8 # number of categories of duration

        initializer = tf.contrib.layers.xavier_initializer() # initializer for fc layers

        RNN_NOTES_CHANNELS    = params['RNN_NOTES_CHANNELS']
        RNN_DURATION_CHANNELS = params['RNN_DURATION_CHANNELS']
        LEARNING_RATE         = params['LEARNING_RATE']

        self.notes = tf.placeholder(tf.float32, (None, None, PITCHES, OCTAVES), 'notes')  # (time, batch, pitch, octave)
        self.durations = tf.placeholder(tf.float32, (None, None, DURATIONS), 'durations')  # (time, batch, duration_category)

        # cut off last note as input, since we have no corresponding output
        # in the case of a single input, don't cut, because we're doing prediction (training makes no sense without an output)
        # TODO: make this more elegant
        with tf.name_scope("trim_inputs"):
            self.notes_in = tf.cond(tf.shape(self.notes)[0] > 1, lambda: self.notes[:-1,:,:,:], lambda: self.notes)
            self.durations_in = tf.cond(tf.shape(self.durations)[0] > 1, lambda: self.durations[:-1,:,:], lambda: self.durations)
        
        # variable to specify how many time steps to use for training each iteration
        self.loss_time_steps = tf.placeholder(tf.int32, name='loss_time_steps')

        # slice off training outputs
        with tf.name_scope("trim_labels"):
            self.notes_labels = self.notes_in[-self.loss_time_steps:, :, :, :]
            self.durations_labels = self.durations_in[-self.loss_time_steps:, :, :]

        # global step counter
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

        # Set variable for dropout of each layer
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Create variables for a couple of useful values
        with tf.name_scope("time_steps"):
            self.times_in   = tf.shape(self.notes_in)[0]
        with tf.name_scope("batch_size"):
            self.batch_size = tf.shape(self.notes_in)[1]

        with tf.name_scope("input_mixer"):
            # take (time, batch, pitch, octave) to (time, batch, duration)
            # followed by a non-linear unit
            # then weighted sum with durations
            # followed by a non-linear unit
            with tf.name_scope("durations_with_notes"):
                W = tf.Variable(initializer((PITCHES, OCTAVES, DURATIONS)), name='W')
                b = tf.Variable(initializer((DURATIONS,)), name='b')
                n2d = tf.tensordot(self.notes_in, W, [[2,3],[0,1]]) + b
                n2d = tf.nn.relu(n2d)
                Wn2d = tf.Variable(initializer((DURATIONS,)), name='Wn2d')
                Wd = tf.Variable(initializer((DURATIONS,)), name='Wd')
                durations_with_notes = Wn2d * n2d + Wd * self.durations_in
                durations_with_notes = tf.nn.relu(durations_with_notes)
            # take (time, batch, duration) to (time, batch, pitch, octave)
            # followed by a non-linear unit
            # then weighted sum with durations
            # followed by a non-linear unit
            with tf.name_scope("notes_with_durations"):
                W = tf.Variable(initializer((DURATIONS, PITCHES, OCTAVES)), name='W')
                b = tf.Variable(initializer((PITCHES, OCTAVES)), name='b')
                d2n = tf.tensordot(self.durations_in, W, [[2],[0]]) + b
                d2n = tf.nn.relu(d2n)
                Wd2n = tf.Variable(initializer((PITCHES, OCTAVES)), name='Wd2n')
                Wn = tf.Variable(initializer((PITCHES, OCTAVES)), name='Wn')
                notes_with_durations = Wd2n * d2n + Wn * self.notes_in
                notes_with_durations = tf.nn.relu(notes_with_durations)

        # create convolutional rnn with notes as input (time, batch, pitch, octave)
        # output (time, batch, pitch, octave)
        with tf.name_scope("rnn_notes") as scope:
            cells = [None] * (len(RNN_NOTES_CHANNELS) + 1)
            cells[0] = Net.conv_cell(2, [PITCHES, OCTAVES, 1], RNN_NOTES_CHANNELS[0], [5,5], self.keep_prob)
            for i in range(1, len(RNN_NOTES_CHANNELS)):
                cells[i] = Net.conv_cell(2, [PITCHES, OCTAVES, RNN_NOTES_CHANNELS[i-1]], RNN_NOTES_CHANNELS[i], [5,5], self.keep_prob)
            cells[-1] = Net.conv_cell(2, [PITCHES, OCTAVES, RNN_NOTES_CHANNELS[-1]], 1, [3,3], self.keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            with tf.name_scope("state"):
                states = Net.get_state_variables(self.batch_size, cell)
            rnn_notes_output, rnn_notes_state = tf.nn.dynamic_rnn(cell, tf.expand_dims(notes_with_durations, -1), initial_state=states, time_major=True, scope=scope)
            rnn_notes_output = tf.squeeze(rnn_notes_output, -1)

        # create rnn with durations as input (time, batch, duration)
        # output (time, batch, duration)
        # end with softmax for duration
        with tf.name_scope("rnn_duration") as scope:
            cell = Net.cell(RNN_DURATION_CHANNELS + [DURATIONS], self.keep_prob)
            with tf.name_scope("state"):
                states = Net.get_state_variables(self.batch_size, cell)
            rnn_duration_output, rnn_duration_state = tf.nn.dynamic_rnn(cell, durations_with_notes, initial_state=states, time_major=True, scope=scope)
            rnn_duration_output = tf.nn.softmax(rnn_duration_output, -1)

        # create tuple of rnn states
        self.rnn_state = rnn_notes_state + rnn_duration_state

        # outputs
        with tf.name_scope("output_loss_time_steps"):
            self.note_out = rnn_notes_output[-self.loss_time_steps:,:,:,:]
            self.duration_out = rnn_duration_output[-self.loss_time_steps:,:,:]

        # compute elementwise L2 norm
        with tf.name_scope("error"):
            error = tf.reduce_mean(tf.square(self.notes_labels - self.note_out)) + tf.reduce_mean(tf.square(self.durations_labels - self.duration_out))

        # optimize
        self.train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error, global_step=self._global_step)

        # assuming that absolute difference between output and correct answer is 0.5
        # or less we can round it to the correct output.
        with tf.name_scope("note_accuracy"):
            accuracy = 1 - tf.reduce_mean(tf.abs(self.notes_labels - tf.cast(self.note_out > 0.5, tf.float32)))
        with tf.name_scope("duration_accuracy"):
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

    @staticmethod
    def cell(num_hidden, keep_prob, activation=tf.nn.relu):
        # create cell definition
        cells = [None] * len(num_hidden)
        for i in range(len(num_hidden)):
            cells[i] = tf.nn.rnn_cell.BasicLSTMCell(num_hidden[i], state_is_tuple=True, activation=activation)
            cells[i] = tf.nn.rnn_cell.DropoutWrapper(cells[i], output_keep_prob=keep_prob)
        return tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

    @staticmethod
    def conv_cell(conv_ndims, input_shape, output_channels, kernel_shape, keep_prob):
        return tf.nn.rnn_cell.DropoutWrapper(
                tf.contrib.rnn.ConvLSTMCell(conv_ndims, input_shape, output_channels, kernel_shape),
                output_keep_prob=keep_prob)

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
            feed_dict[self.rnn_state] = batch_state

        return self.session.run(
            [self.train_fn, self.rnn_state],
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
            feed_dict[self.rnn_state] = batch_state

        return self.session.run(
            [self.note_out, self.duration_out, self.rnn_state],
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
            # make trainable initial variable which DOES NOT include batch size, but tile so that each batch gets it
            state_init_c = tf.tile(tf.Variable(tf.zeros([1] + state_c.shape[1:].as_list())), [batch_size] + (state_c.shape.ndims - 1) * [1])
            state_init_h = tf.tile(tf.Variable(tf.zeros([1] + state_h.shape[1:].as_list())), [batch_size] + (state_h.shape.ndims - 1) * [1])
            state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                tf.placeholder_with_default(state_init_c, state_c.shape, "State_C"),
                tf.placeholder_with_default(state_init_h, state_h.shape, "State_H")))
                #tf.Variable(state_c, trainable=False),
                #tf.Variable(state_h, trainable=False)))
        # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
        return tuple(state_variables)
