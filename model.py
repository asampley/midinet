import tensorflow as tf

class Net:
    def __init__(self, session, params):
        self.session = session

        initializer = tf.contrib.layers.xavier_initializer() # initializer for fc layers

        RNN_SIZES     = params['RNN_SIZES']
        LEARNING_RATE = params['LEARNING_RATE']
        DATA_SIZES    = params['DATA_SIZES']
        DATA_NAMES    = params['DATA_NAMES']

        assert(len(DATA_SIZES) == len(DATA_NAMES))
        DATA_ELEMENTS = len(DATA_SIZES)

        self.messages = tf.placeholder(tf.int32, (None, None, DATA_ELEMENTS), 'messages') # (time, batch, message)
        
        with tf.name_scope("one_hot_stitch"):
            # slice and create one_hot vectors for each variable (pitch, octave, volume, duration)
            one_hots = []
            for i in range(DATA_ELEMENTS):
                ds = DATA_SIZES[i]
                name = DATA_NAMES[i] + '_one_hot'
                one_hots += [tf.one_hot(self.messages[:,:,i], ds, name=name)]

        # cut off last note as input, since we have no corresponding output
        # in the case of a single input, don't cut, because we're doing prediction (training makes no sense without an output)
        # TODO: make this more elegant
        with tf.name_scope("trim_inputs"):
            # join each one_hot vector into a longer vector of size (time, batch, pitch + octave + volume + duration)
            self.input = tf.concat(one_hots, -1)
            self.input = tf.cond(tf.shape(self.input)[0] > 1, lambda: self.input[:-1, ...], lambda: self.input)
        
        # variable to specify how many time steps to use for training each iteration
        self.loss_time_steps = tf.placeholder(tf.int32, name='loss_time_steps')

        # slice off training outputs
        with tf.name_scope("trim_labels"):
            self.label_slices = [None] * DATA_ELEMENTS
            for i in range(DATA_ELEMENTS):
                self.label_slices[i] = one_hots[i][-self.loss_time_steps:, ...]
            self.labels = tf.concat(self.label_slices, -1)

        # global step counter
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

        # Set variable for dropout of each layer
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Create variables for a couple of useful values
        with tf.name_scope("time_steps"):
            self.times_in   = tf.shape(self.messages)[0]
        with tf.name_scope("batch_size"):
            self.batch_size = tf.shape(self.messages)[1]

        # create RNNs for each layer, followed by a final one that produces a one_hot the same size as the stitched input
        # input size:  (time, batch, pitch + octave + volume + duration)
        # output size: (time, batch, pitch + octave + volume + duration)
        with tf.name_scope("rnn") as scope:
            # create RNN cell, which is multiple lstm cells
            cell = Net.cell(RNN_SIZES + [sum(DATA_SIZES)], self.keep_prob)
            # create trainable initial state
            with tf.name_scope("state"):
                states = Net.get_state_variables(self.batch_size, cell)
            # create the whole RNN
            rnn_output, self.rnn_state = tf.nn.dynamic_rnn(cell, self.input, initial_state=states, time_major=True, scope=scope)
        
        # output only enough time steps for loss
        with tf.name_scope("output_loss_time_steps"):
            rnn_output = rnn_output[-self.loss_time_steps:, ...]

        # slice and do softmax for each one_hot vector
        with tf.name_scope("output_softmax"):
            splits = tf.split(rnn_output, DATA_SIZES, 2)
            self.output_slices = [None] * len(DATA_SIZES)
            for i in range(len(DATA_SIZES)):
                name = DATA_NAMES[i] + '_softmax'
                self.output_slices[i] = tf.nn.softmax(splits[i], -1, name=name)
            # rejoin after softmax into one vector
            self.output = tf.concat(self.output_slices, -1)

        # compute elementwise L2 norm
        with tf.name_scope("error"):
            error = tf.reduce_mean(tf.square(self.labels - self.output))

        # optimize
        self.train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error, global_step=self._global_step)

        # compute accuracies as fraction of outputs where the value with the maximal value is considered the chosen one
        with tf.name_scope("accuracy"):
            accuracy = [None] * DATA_ELEMENTS
            for i in range(DATA_ELEMENTS):
                accuracy[i] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output_slices[i], axis=-1), tf.argmax(self.label_slices[i], axis=-1)), tf.float32))
        
        # Make summary op and file
        with tf.name_scope("summaries"):
            tf.summary.scalar('error', error)
            for i in range(DATA_ELEMENTS):
                name = DATA_NAMES[i]
                tf.summary.scalar(name + '_accuracy', accuracy[i])
                tf.summary.histogram(name + '_labels', tf.argmax(self.label_slices[i], axis=-1))
                tf.summary.histogram(name + '_output', tf.argmax(self.output_slices[i], axis=-1))

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

    def save(self):
        self.saver.save(self.session, 'model/model.ckpt')

    def restore(self):
        self.saver.restore(self.session, 'model/model.ckpt')
    
    def train(self, messages, loss_time_steps, batch_state = None, keep_prob = 0.5):
        feed_dict = {
            self.messages: messages,
            self.loss_time_steps: loss_time_steps,
            self.keep_prob: keep_prob
        }
        if batch_state is not None:
            feed_dict[self.rnn_state] = batch_state

        return self.session.run(
            [self.train_fn, self.rnn_state],
            feed_dict=feed_dict)

    def predict(self, messages, batch_state = None):
        """
        Returns the outputs and new state of the lstm
        """
        
        feed_dict = {
            self.messages: messages,
            self.loss_time_steps: 2,
            self.keep_prob: 1.0
        }
        if batch_state is not None:
            feed_dict[self.rnn_state] = batch_state

        return self.session.run(
            [self.output_slices, self.rnn_state],
            feed_dict=feed_dict)

    def summarize(self, messages, loss_time_steps, batch_state = None):
        feed_dict = {
            self.messages: messages,
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
