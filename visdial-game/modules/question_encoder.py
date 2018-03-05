import tensorflow as tf

class QuestionEncoder(object):
	"""
		Takes in the previous question and runs each question through an LSTM.
		### Dimensions
		Questions: float (batch_size, max_sequence_length, vocabulary_size)
		Encoded Questions: float (batch_size, hidden_dimension)
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
		Builds the graph to take in the questions and ouput embeddings for each question.
		===================================
		"""
		with tf.varible_scope(self.scope):
            # Stack RNN Cells: input -> basic rnn cell -> basic lstm cell -> output
			cells = [tf.contrib.BasicRNNCell(self.hidden_dimension), tf.contrib.BasicLSTMCell(self.hidden_dimension)]
			self.cell = tf.contrib.rnn.MultiRNNCell(cells)
    
    def encode_questions(self, questions):
		"""
		Given a question, output an embedding for the question.
		===================================
		INPUTS:
		questions: float - (batch_size, max_sequence_length, vocabulary_size)
		"""
		batch_size = tf.shape(questions)[0]
		initial_state = self.cell.zero_state(batch_size, dtype=tf.float32)
		encoded_questions, _ = tf.nn.dynamic_rnn(
			self.rnn,
			questions,
			dtype=tf.float32
		)