import tensorflow as tf
import os

class Net:
    def __init__(self, session, params):
        self.session = session

        initializer = tf.contrib.layers.xavier_initializer() # initializer for fc layers

        RNN_SIZES     = params['RNN_SIZES']     # tuple:                    containing the output size of each rnn layer
        LEARNING_RATE = params['LEARNING_RATE'] # float:                    learning rate for sgd
        DATA_SIZES    = params['DATA_SIZES']    # list or tuple of ints:    number of possible values for each data element
        DATA_NAMES    = params['DATA_NAMES']    # list or tuple of strings: name for each data element
        DATA_WEIGHTS  = params['DATA_WEIGHTS']  # list or tuple of floats:  weight for each data element in total loss
        self.SAVE_DIR = params['SAVE_DIR']      # string:                   directory to save summaries and the neural network
        CHANNEL_DIM   = params['CHANNEL_DIM']   # int:                      dimension which is not included in the convolution kernel
        KERNEL_SIZE   = params['KERNEL_SIZE']   # list:                     size of kernel for convolutions

        assert(len(DATA_SIZES) == len(DATA_NAMES) and len(DATA_NAMES) == len(DATA_WEIGHTS))
        DATA_ELEMENTS = len(DATA_SIZES)
        CONV_NDIMS    = DATA_ELEMENTS - 1

        self.messages = tf.placeholder(tf.int32, (None, None, DATA_ELEMENTS), 'messages') # (time, batch, message)
        # this hot mess creates a tensor with all the notes represented as a single 1 in a matrix of size DATA_ELEMENTS
        messages_flat = Net.collapse_dims(self.messages, 0, 2)
        self.one_hot = tf.stack(tf.map_fn(
                lambda t: tf.sparse_to_dense( # (time * batch) + DATA_SIZES
                    tf.expand_dims(t, 0), DATA_SIZES, 1),
                messages_flat))
        self.one_hot = tf.reshape( # (time, batch) + DATA_SIZES
                self.one_hot,
                tf.concat((tf.shape(self.messages)[:2], DATA_SIZES), 0))

        # convert one_hot to float32
        self.one_hot = tf.cast(self.one_hot, tf.float32)
        self.weights = tf.constant(DATA_WEIGHTS, name='weights')
        
        # cut off last note as input, since we have no corresponding output
        # in the case of a single input, don't cut, because we're doing prediction (training makes no sense without an output)
        # TODO: make this more elegant
        with tf.name_scope("trim_inputs"):
            # reorder one_hot to put the specified channel dim as the last dimension
            self.input = tf.transpose(self.one_hot, list(range(2 + CHANNEL_DIM)) + list(range(2 + CHANNEL_DIM + 1, 2 + DATA_ELEMENTS)) + [CHANNEL_DIM + 2])
            self.input = tf.cond(tf.shape(self.input)[0] > 1, lambda: self.input[:-1, ...], lambda: self.input)
        
        # variable to specify how many time steps to use for training each iteration
        self.loss_time_steps = tf.placeholder(tf.int32, name='loss_time_steps')

        # slice off training outputs
        with tf.name_scope("trim_labels"):
            self.label = self.one_hot[-self.loss_time_steps:, ...]
        # flatten labels into (time * batch, one_hot)
        with tf.name_scope("flatten_labels"):
            self.label_flat = Net.collapse_dims(Net.collapse_dims(self.label, 2), 0, 2)

        # global step counter
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

        # Set variable for dropout of each layer
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Create variables for a couple of useful values
        with tf.name_scope("time_steps"):
            self.times_in   = tf.shape(self.messages)[0]
        with tf.name_scope("batch_size"):
            self.batch_size = tf.shape(self.messages)[1]

        # create RNNs for each layer, followed by a final one that produces a shape the same size as the input
        # input size:  (time, batch) + CONV_DIMS + (CHANNEL_DIM)
        # output size: (time, batch) + CONV_DIMS + (CHANNEL_DIM)
        with tf.name_scope("rnn") as scope:
            # create RNN cell
            cell = Net.cell(CONV_NDIMS, self.input.shape[2:].as_list(), RNN_SIZES + (self.input.shape.as_list()[-1],), KERNEL_SIZE, self.keep_prob)

            # create trainable initial state
            with tf.name_scope("state"):
                states = Net.get_state_variables(self.batch_size, cell)
            # create the whole RNN
            rnn_output, self.rnn_state = tf.nn.dynamic_rnn(cell, self.input, initial_state=states, time_major=True, scope=scope)
        
        # output only enough time steps for loss
        with tf.name_scope("output_loss_time_steps"):
            rnn_output = rnn_output[-self.loss_time_steps:, ...]

        # flatten outputs into (time * batch, one_hot)
        with tf.name_scope("output_flat"):
            self.output_flat = Net.collapse_dims(Net.collapse_dims(rnn_output, 2), 0, 2)

        # output as probabilities in matrix
        with tf.name_scope("output"):
            self.output = tf.nn.softmax(self.output_flat, -1)
            self.output = tf.reshape(self.output, tf.shape(rnn_output))
            # reorder to input order
            self.output = tf.transpose(self.output, list(range(2 + CHANNEL_DIM)) + [2 + DATA_ELEMENTS - 1] + list(range(2 + CHANNEL_DIM, 2 + DATA_ELEMENTS - 1))) 

        # compute cross entropy
        with tf.name_scope("error"):
            error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_flat, logits=self.output_flat))

        # optimize
        self.train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error, global_step=self._global_step)

        # compute accuracies as fraction of outputs where the value with the maximal value is considered the chosen one
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output_flat, axis=-1), tf.argmax(self.label_flat, axis=-1)), tf.float32))
            accuracies = [None] * DATA_ELEMENTS
            for i in range(DATA_ELEMENTS):
                with tf.name_scope(DATA_NAMES[i]):
                    # note that argmax can only be called on rank <=5 tensors, so dimensions must be merged first
                    collapsed_output = Net.collapse_dims(Net.collapse_dims(self.output, 0, i+2), 2)
                    collapsed_label  = Net.collapse_dims(Net.collapse_dims(self.label, 0, i+2), 2)
                    accuracies[i] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(collapsed_output, axis=1), tf.argmax(collapsed_label, axis=1)), tf.float32))
        
        # Make summary op and file
        with tf.name_scope('summary'):
            tf.summary.scalar('error', error)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.histogram('label_flat', tf.argmax(self.label_flat, -1))
            tf.summary.histogram('output_flat', tf.argmax(self.output_flat, -1))
            for i in range(DATA_ELEMENTS):
                name = DATA_NAMES[i]
                tf.summary.scalar('accuracy_' + name, accuracies[i])
            self.summaries = tf.summary.merge_all()
            self.summaryFileWriter = tf.summary.FileWriter(self.SAVE_DIR, self.session.graph)

        # Make net saver
        self.saver = tf.train.Saver()

    @staticmethod
    def cell(conv_ndims, input_shape, output_channels, kernel_shape, keep_prob):
        """
        Creates N ConvLSTMCells each with DropoutWrappers
        conv_ndims: int
        input_shape: shape of input tensor conv_dims + (input_channels)
        output_channels: list of output channels for each LSTM
        kernel_shape: shape of kernel
        """
        LAYERS = len(output_channels)
        input_shape_base = input_shape[:-1]
        kernel_shape = list(kernel_shape)

        cells = [None] * LAYERS
        cells[0] = tf.contrib.rnn.ConvLSTMCell(conv_ndims, input_shape, output_channels[0], kernel_shape)
        for i in range(1, LAYERS):
            cells[i] = tf.contrib.rnn.ConvLSTMCell(conv_ndims, input_shape_base + [output_channels[i-1]], output_channels[i], kernel_shape)
            cells[i] = tf.nn.rnn_cell.DropoutWrapper(cells[i], output_keep_prob=keep_prob)
        return tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

    def save(self):
        self.saver.save(self.session, os.path.join(self.SAVE_DIR, 'model.ckpt'))

    def restore(self):
        self.saver.restore(self.session, os.path.join(self.SAVE_DIR + 'model.ckpt'))
    
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
            [self.output, self.rnn_state],
            feed_dict=feed_dict)

    def summarize(self, messages, loss_time_steps, batch_state = None):
        feed_dict = {
            self.messages: messages,
            self.loss_time_steps: loss_time_steps,
            self.keep_prob: 1.0
        }
        if batch_state is not None:
            feed_dict[self.rnn_state] = batch_state

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
            state_init_c = tf.tile(tf.Variable(tf.zeros([1] + state_c.shape[1:].as_list())), [batch_size] + ((state_c.shape.ndims - 1) * [1]))
            state_init_h = tf.tile(tf.Variable(tf.zeros([1] + state_h.shape[1:].as_list())), [batch_size] + ((state_h.shape.ndims - 1) * [1]))
            state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                tf.placeholder_with_default(state_init_c, state_c.shape, "State_C"),
                tf.placeholder_with_default(state_init_h, state_h.shape, "State_H")))
                #tf.Variable(state_c, trainable=False),
                #tf.Variable(state_h, trainable=False)))
        # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
        return tuple(state_variables)

    @staticmethod
    def collapse_dims(tensor, start, end=None):
        """
        Collapse a range of dimensions [start, end) in tensor into a single dimension
        This operation can be undone by using tf.reshape with the original shape.

        If no end is specified, then all dims from start onwards are collapsed

        If end == start, then nothing is done
        """
        if end is None:
            end = tensor.shape.ndims
        
        if start == end: return tensor
        with tf.name_scope('collapse_dims'):
            return tf.reshape(tensor, tf.concat((tf.shape(tensor)[:start], tf.reduce_prod(tf.shape(tensor)[start:end], keep_dims=True), tf.shape(tensor)[end:]), 0))

