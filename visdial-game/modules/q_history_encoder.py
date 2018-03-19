import tensorflow as tf

class QHistoryEncoder(object):
    """
        Takes in previous state, and returns the question for that time step
        ### Dimensions
        Previous state: (batch_size, hidden_dimension)
    """
    def __init__(self, hidden_dimension, scope):
        """
        Initialization function
        ====================
        INPUTS:
        hidden_dimension: int - shape of the hidden state for the LSTM/RNN cells used
        ====================
        """
        self.hidden_dimension = hidden_dimension
        self.scope = scope
        self.add_cells()

    def add_cells(self):
        """
        Creates the RNN's which do the dirty work
        ===================================
        """
        with tf.variable_scope(self.scope):
            self.cell = tf.contrib.rnn.GRUCell(self.hidden_dimension)

    def generate_next_state(self, current_facts, prev_states=None):
        """
        Builds the graph to take in the previous state, and current fact and generates the new state
        ===================================
        INPUTS:
        current_facts: float of shape (batch_size, hidden_dimension) - The encoding of the (q,a) pair for this round of dialog
        prev_states: float of shape (batch_size, hidden_dimension) - The state/history encoding for this round of dialog
        ===================================
        OUTPUTS:
            outputs: float of shape (batch_size, max_fact_length, hidden_dimension)
            next_states: float of shape (batch_size, hidden_dimension) - The new state/history encoding generated using the current_fact
        """
        if prev_states is None:
            prev_states = self.cell.zero_state(tf.shape(current_facts)[0], dtype=tf.float32)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            _, next_states = self.cell(
                current_facts,
                prev_states
            )
            return next_states
