import tensorflow as tf

class AHistoryEncoder(object):
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

	def generate_next_state(self, prev_states, current_facts, questions, images):
		"""
		Builds the graph to take in the previous state, and current fact and generates the new state
		===================================
		INPUTS:
		prev_states: float of shape (batch_size, hidden_dimension) - The state/history encoding for this round of dialog
		current_facts: float of shape (batch_size, hidden_dimension) - The encoding of the (q,a) pair for the past round of dialog
        questions: float of shape (batch_size, hidden_dimension) - The encoding of the question for the current round of dialog
        images: float of shape (batch_size, vgg_dim) - The encoding of the images for the current round of dialog
		===================================
		OUTPUTS:
			outputs: float of shape (batch_size, hidden_dimension + hidden_dimension + vgg_dim, hidden_dimension)
			next_states: float of shape (batch_size, hidden_dimension) - The new state/history encoding generated using the current_fact
		"""
        inputs = tf.concat([questions, images, current_facts], 1)
		with tf.varible_scope(self.scope):
			outputs, next_states = tf.nn.dynamic_rnn(
				self.cell,
				inputs,
				initial_state=prev_states,
				dtype=tf.float32,
			)
			return outputs ,next_states