import tensorflow as tf
import os

class Net:
    def __init__(self, session, params):
        self.session = session

        initializer = tf.contrib.layers.xavier_initializer() # initializer for fc layers

        RNN_SIZES     = params['RNN_SIZES']     # list or tuple: containing the output size of each rnn layer
        DENSE_SIZES   = params['DENSE_SIZES']   # list or tuple: containing the output size of each dense layer
        LEARNING_RATE = params['LEARNING_RATE'] # float:         learning rate for sgd
        CATEGORIES    = params['CATEGORIES']    # int:           number of unique notes in the data
        self.SAVE_DIR = params['SAVE_DIR']      # string:        directory to save summaries and the neural network

        # data, both input and labels
        self.indices = tf.placeholder(tf.int32, (None, None), 'message_indices') # (time, batch)
        self.one_hots = tf.one_hot(self.indices, CATEGORIES, dtype=tf.float32, name='one_hots') # (time, batch, one_hot)
        # boolean scalar, for whether to trim off the last datum. Should be true for training, false for prediction
        self.trim_last = tf.placeholder(tf.bool, (), 'trim_last') # scalar
        # Set variable for dropout of each layer
        self.keep_prob = tf.placeholder(tf.float32, (), name='keep_prob') # scalar
        
        # cut off last note as input, since we have no corresponding output
        # only does this if self.trim_last is true
        with tf.name_scope("trim_inputs"):
            # join each one_hot vector into a longer vector of size (time, batch, pitch + octave + volume + duration)
            self.input = tf.cond(self.trim_last, lambda: self.one_hots[:-1, ...], lambda: self.one_hots)
        
        # variable to specify how many time steps to use for training each iteration
        self.loss_time_steps = tf.placeholder(tf.int32, name='loss_time_steps')

        # slice off labels
        with tf.name_scope("trim_labels"):
            self.labels = self.one_hots[-self.loss_time_steps:, ...]

        # global step counter
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

        # Create useful variables
        with tf.name_scope("batch_size"):
            self.batch_size = tf.shape(self.indices)[1]

        # create RNNs for each layer
        # input size:  (time, batch, categories)
        # output size: (time, batch, RNN_SIZES[-1])
        with tf.name_scope("rnn") as scope:
            # create RNN cell, which is multiple lstm cells
            cell = Net.cell(RNN_SIZES, self.keep_prob)
            # create trainable initial state
            with tf.name_scope("state"):
                states = Net.get_state_variables(self.batch_size, cell)
            # create the whole RNN
            rnn_output, self.rnn_state = tf.nn.dynamic_rnn(cell, self.input, initial_state=states, time_major=True, scope=scope)
 
        # output only enough time steps for loss
        with tf.name_scope("output_loss_time_steps"):
            rnn_output = rnn_output[-self.loss_time_steps:, ...]

        # create dense layers, followed by a final one that produces the correct number of categories
        with tf.variable_scope("dense"):
            in_tensor = rnn_output
            for i in range(len(DENSE_SIZES)):
                with tf.variable_scope("layer_" + str(i)):
                    in_tensor = tf.layers.dense(in_tensor, DENSE_SIZES[i], activation=tf.nn.relu)
            with tf.variable_scope("layer_" + str(len(DENSE_SIZES))):
                self.output_raw = tf.layers.dense(in_tensor, CATEGORIES, activation=tf.nn.relu)

        # do softmax
        with tf.name_scope("softmax"):
            self.output = tf.nn.softmax(self.output_raw)

        # compute cross entropy
        with tf.name_scope("error"):
            self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output_raw, name='error'))

        # optimize
        self.train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.error, global_step=self._global_step)

        # compute accuracies as fraction of outputs where the value with the maximal value is considered the chosen one
        with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output, axis=-1), tf.argmax(self.labels, axis=-1)), tf.float32))
        
        # Make summary op and file
        with tf.name_scope('summary'):
            tf.summary.scalar('error', self.error)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('label', tf.argmax(self.labels, axis=-1))
            tf.summary.histogram('output', tf.argmax(self.output, axis=-1))

            self.summaries = tf.summary.merge_all()
            self.summaryFileWriter = tf.summary.FileWriter(self.SAVE_DIR, self.session.graph)

        # Make net saver
        self.saver = tf.train.Saver()

    @staticmethod
    def cell(num_hidden, keep_prob, activation=tf.tanh):
        # create cell definition
        cells = [None] * len(num_hidden)
        for i in range(len(num_hidden)):
            cells[i] = tf.nn.rnn_cell.BasicLSTMCell(num_hidden[i], state_is_tuple=True, activation=activation)
            cells[i] = tf.nn.rnn_cell.DropoutWrapper(cells[i], output_keep_prob=keep_prob)
        return tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

    def save(self):
        self.saver.save(self.session, os.path.join(self.SAVE_DIR + 'model.ckpt'))

    def restore(self):
        self.saver.restore(self.session, os.path.join(self.SAVE_DIR, 'model.ckpt'))
    
    def train(self, indices, loss_time_steps, batch_state = None, keep_prob = 0.5):
        feed_dict = {
            self.indices: indices,
            self.loss_time_steps: loss_time_steps,
            self.keep_prob: keep_prob,
            self.trim_last: True
        }
        if batch_state is not None:
            feed_dict[self.rnn_state] = batch_state

        return self.session.run(
            [self.train_fn, self.rnn_state],
            feed_dict=feed_dict)

    def predict(self, indices, batch_state = None):
        """
        Returns the outputs and new state of the lstm
        """
        
        feed_dict = {
            self.indices: indices,
            self.loss_time_steps: 1,
            self.keep_prob: 1.0,
            self.trim_last: False
        }
        if batch_state is not None:
            feed_dict[self.rnn_state] = batch_state

        return self.session.run(
            [self.output, self.rnn_state],
            feed_dict=feed_dict)

    def summarize(self, indices, loss_time_steps, batch_state = None):
        feed_dict = {
            self.indices: indices,
            self.loss_time_steps: loss_time_steps,
            self.keep_prob: 1.0,
            self.trim_last: True
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
