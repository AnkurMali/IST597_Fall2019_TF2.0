# code is based on official tensorflow repo
# codebase is identical to the original math present in original LSTM paper
# Author :- aam35

class BasicLSTM_cell(object):

    def __init__(self, input_units, hidden_units, output_units):
    
        # Now you we go basic based on our previous all assignments.
        # Let's understand how lstm is created from scratch

        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        
        # Declare all weight and biases for lstm cell , these are trainable variables for you.
        # Pass these trainable variables to gradient Tape to calculate updates
        
        
        # Declare weights for input gate

        self.Wi = tf.Variable(tf.zeros([self.input_units, self.hidden_units]))
        self.Ui = tf.Variable(tf.zeros([self.hidden_units, self.hidden_units]))
        self.bi = tf.Variable(tf.zeros([self.hidden_units]))
        
        # Declare weights for forget gate

        self.Wf = tf.Variable(tf.zeros([self.input_units, self.hidden_units]))
        self.Uf = tf.Variable(tf.zeros([self.hidden_units, self.hidden_units]))
        self.bf = tf.Variable(tf.zeros([self.hidden_units]))
        
        # Declare weights for output gate

        self.Woutg = tf.Variable(tf.zeros([self.input_units, self.hidden_units]))
        self.Uoutg = tf.Variable(tf.zeros([self.hidden_units, self.hidden_units]))
        self.boutg = tf.Variable(tf.zeros([self.hidden_units]))
        
        # Declare weights for cell state

        self.Wc = tf.Variable(tf.zeros([self.input_units, self.hidden_units]))
        self.Uc = tf.Variable(tf.zeros([self.hidden_units, self.hidden_units]))
        self.bc = tf.Variable(tf.zeros([self.hidden_units]))

        # Weights for output layers
        
        self.Wo = tf.Variable(tf.truncated_normal([self.hidden_units, self.output_units], mean=0, stddev=.02))
        self.bo = tf.Variable(tf.truncated_normal([self.output_units], mean=0, stddev=.02))

        # Placeholder for input vector with shape[batch_size, seq, embeddings]
        
        self._inputs = tf.placeholder(tf.float32, shape=[None, None, self.input_nodes], name='inputs')

       
        batch_input_ = tf.transpose(self._inputs, perm=[2, 0, 1])
        self.processed_input = tf.transpose(batch_input_)

        self.initial_hidden = self._inputs[:, 0, :]
        self.initial_hidden = tf.matmul(self.initial_hidden, tf.zeros([input_nodes, hidden_unit]))

        self.initial_hidden = tf.stack([self.initial_hidden, self.initial_hidden])

    def Lstm(self, previous_hidden_memory, x):
        # previous_hidden_memory = [previous_hidden_state, prevous memory]
        # x is input
        # return hidden_state

        previous_hidden_state, c_prev = tf.unstack(previous_hidden_memory)

        i = tf.sigmoid( tf.matmul(x, self.Wi) +
                        tf.matmul(previous_hidden_state, self.Ui) + self.bi)

        f = tf.sigmoid( tf.matmul(x, self.Wf) +
                        tf.matmul(previous_hidden_state, self.Uf) + self.bf)

        o = tf.sigmoid( tf.matmul(x, self.Woutg) +
                        tf.matmul(previous_hidden_state, self.Uoutg) + self.boutg)

        c_ = tf.nn.tanh(tf.matmul(x, self.Wc) +
                        tf.matmul(previous_hidden_state, self.Uc) + self.bc)

        
        c = f * c_prev + i * c_
        current_hidden_state = o * tf.nn.tanh(c)

        return tf.stack([current_hidden_state, c])

    def get_states(self):
        all_hidden_states = tf.scan(self.Lstm, self.processed_input, initializer=self.initial_hidden, name='states')
        all_hidden_states = all_hidden_states[:, 0, :, :]
        return all_hidden_states

    def get_output(self, hidden_state):
        output = tf.nn.relu(tf.matmul(hidden_state, self.Wo) + self.bo)
        return output

    def get_outputs(self):
        all_hidden_states = self.get_states()
        all_outputs = tf.map_fn(self.get_output, all_hidden_states)
        return all_outputs
