class fact_encoder():

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
		TODO: Differentiate between supervised pre-training and RL-training:
		===================================
		"""
		with tf.varible_scope(self.scope):

			cells = [tf.contrib.BasicRNNCell(self.hidden_dimension), tf.contrib.BasicLSTMCell(self.hidden_dimension)]
			#stacked cell

			self.cell = tf.contrib.rnn.MultiRNNCell(cells)

	def generate_next_fact(self, question, answer):
		"""
		Builds the graph to take in the question, answer and generates the new fact
		===================================
		INPUTS:
		question: float of shape (batch_size, max_question_length, embedding_dimension) - The question from this round of dialog
		answer: float of shape (batch_size, max_question_length, embedding_dimension) - The answer from this round of dialog
		===================================
		OUTPUTS:
			next_fact: float of shape (batch_size, hidden_dimension) - The new fact encoding generated using the current answer and question
		"""

		with tf.varible_scope(self.scope):
			#start_tokens to 
			# start_tokens = tf.tile(self.start_token, [tf.shape(states)[0],1])
			inputs = tf.concat([question, answer], 2)
			_, next_state = self.cell(inputs, tf.zeros([tf.shape(question)[0], self.hidden_dimension], dtype = tf.float32))
			
			return next_fact
