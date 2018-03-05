class Q_history_encoder():
	"""
		Takes in previous state, and returns the question for that time step
		###Dimensions
		Previous state: (batch_size, hidden_dimension)
		Question: (batch_size, question_length, indices)
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
		TODO: Differentiate between supervised pre-training and RL-training:
		===================================
		"""
		with tf.varible_scope(self.scope):

			cells = [tf.contrib.BasicRNNCell(self.hidden_dimension), tf.contrib.BasicLSTMCell(self.hidden_dimension)]
			#stacked cell

			self.cell = tf.contrib.rnn.MultiRNNCell(cells)

	def generate_next_state(self, prev_state, current_fact):
		"""
		Builds the graph to take in the previous state, and current fact and generates the new state
		===================================
		INPUTS:
		prev_state: float of shape (batch_size, hidden_dimension) - The state/history encoding for this round of dialog
		current_fact: float of shape (batch_size, hidden_dimension) - The encoding of the (q,a) pair for this round of dialog
		===================================
		OUTPUTS:
			next_state: float of shape (batch_size, hidden_dimension) - The new state/history encoding generated using the current_fact
		"""

		with tf.varible_scope(self.scope):
			#start_tokens to 
			# start_tokens = tf.tile(self.start_token, [tf.shape(states)[0],1])
			
			output, next_state = self.cell(current_fact, prev_state)
			return output,next_state
