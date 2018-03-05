import tensorflow as tf

class QHistoryEncoder(object):
	"""
		Takes in previous state, and returns the question for that time step
		### Dimensions
		Previous state: (batch_size, hidden_dimension)
		Question: (batch_size, question_length, indices)
	"""
	def __init__(self, hidden_dimension, scope):
		"""
		Initialization function
		====================
		INPUTS:
		hidden_dimension: int - shape of the hidden state for the LSTM/RNN cells used
		start_token : float of shape (embedding_size) - word embedding of the start_token
		max_question_length : int - length of longest question
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
		with tf.varible_scope(self.scope):
			cells = [tf.contrib.BasicRNNCell(self.hidden_dimension), tf.contrib.BasicLSTMCell(self.hidden_dimension)]
			self.cell = tf.contrib.rnn.MultiRNNCell(cells)

	def generate_next_state(self, prev_state, current_facts):
		"""
		Builds the graph to take in the previous state, and current fact and generates the new state
		===================================
		INPUTS:
		prev_state: float of shape (batch_size, hidden_dimension) - The state/history encoding for this round of dialog
		current_facts: float of shape (batch_size, hidden_dimension) - The encoding of the (q,a) pair for this round of dialog
		===================================
		OUTPUTS:
			outputs: float of shape (batch_size, max_fact_length, hidden_dimension)
			next_states: float of shape (batch_size, hidden_dimension) - The new state/history encoding generated using the current_fact
		"""
		with tf.varible_scope(self.scope):
			outputs, next_states = tf.nn.dynamic_rnn(
				self.cell,
				current_facts,
				initial_state=prev_state,
				sequence_length=None, # TODO: Add this maybe?
				dtype=tf.float32,
			)
			return outputs ,next_states
