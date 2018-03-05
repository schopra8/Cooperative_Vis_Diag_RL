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
		with tf.varible_scope(self.scope):
			cells = [tf.contrib.BasicRNNCell(self.hidden_dimension), tf.contrib.BasicLSTMCell(self.hidden_dimension)]
			self.cell = tf.contrib.rnn.MultiRNNCell(cells)

	def generate_fact_from_captions(self, captions, caption_lengths):
		"""
		Builds the graph to take in the questions, answers and generates the new fact
		===================================
		INPUTS:
		captions: float of shape (batch_size, caption_size, embedding_dimension) - Captions
		===================================
		OUTPUTS:
			next_facts: float of shape (batch_size, hidden_dimension) - The new fact encoding generated using the current answers and question
		"""
		with tf.varible_scope(self.scope):
			_, next_facts = tf.nn.dynamic_rnn(
				self.cell,
				inputs,
				sequence_length=, # TODO: Add this maybe?
				dtype=tf.float32,
			)
			return next_facts

	def generate_next_fact(self, questions, answers):
		"""
		Builds the graph to take in the questions, answers and generates the new fact
		===================================
		INPUTS:
		questions: float of shape (batch_size, max_question_length, embedding_dimension) - The questions from this round of dialog
		answers: float of shape (batch_size, max_answer_length, embedding_dimension) - The answers from this round of dialog
		===================================
		OUTPUTS:
			next_facts: float of shape (batch_size, hidden_dimension) - The new fact encoding generated using the current answers and question
		"""
		with tf.varible_scope(self.scope):
			# TODO: Determine how to extract questions and answers (removing padding for questions and answers)
			inputs = tf.concat([questions, answers], 1) # Concatenate along max_question_length/max_answer_length dimension
			_, next_facts = tf.nn.dynamic_rnn(
				self.cell,
				inputs,
				sequence_length=None, # TODO: Add this maybe?
				dtype=tf.float32,
			)
			return next_facts
