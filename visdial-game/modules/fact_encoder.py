import tensorflow as tf

class FactEncoder(object):
	def __init__(self, hidden_dimension, scope):
		"""
		Initialization function
		====================
		INPUTS:
		hidden_dimension: int - shape of the hidden state for the LSTM/RNN cells used
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
		with tf.variable_scope(self.scope):
			cells = [tf.contrib.rnn.BasicRNNCell(self.hidden_dimension), tf.contrib.rnn.BasicLSTMCell(self.hidden_dimension)]
			self.cell = tf.contrib.rnn.MultiRNNCell(cells)

	def generate_fact(self, inputs, input_lengths):
		"""
		Builds the graph to take in some inputs and generate facts
		===================================
		INPUTS:
		inputs: float of shape (batch_size, max_size)
		input_lengths: float of shape (batch_size)
		===================================
		OUTPUTS:
			next_facts: float of shape (batch_size, hidden_dimension) - The new fact encoding generated using the current answers and question
		"""
		with tf.variable_scope(self.scope):
			_, next_facts = tf.nn.dynamic_rnn(
				self.cell,
				inputs,
				sequence_length=input_lengths, 
				dtype=tf.float32,
			)
			return next_facts
