import tensorflow as tf
from tensorflow.python.layers.core import Dense

class QuestionDecoder(object):
	"""
		Takes in previous state, and returns the question for that time step
		### Dimensions
		Previous state: (batch_size, hidden_dimension)
		Question: (batch_size, question_length, indices)
	"""
	def __init__(self, hidden_dimension, start_token_embedding, end_token_idx, max_question_length, vocabulary_size, embedding_lookup, scope):
		"""
		Initialization function
		====================
		INPUTS:
		hidden_dimension: int - shape of the hidden state for the LSTM/RNN cells used
		start_token_embedding : float of shape (embedding_size) - word embedding of the start_token_embedding
		end_token_idx: int of index referring to the end token within our vocabulary
		max_question_length : int - length of longest question
		vocabulary_size: int - size of vocabulary (including start and end tokens)
		embedding_lookup: callable function which returns an embeddings tensor (batch_size, embedding_size) given a vector of indices into our vocabulary
		scope: variable scope of decoder
		====================
		"""
		self.hidden_dimension = hidden_dimension
		self.start_token_embedding = start_token_embedding
		self.end_token_idx = end_token_idx
		self.max_question_length = max_question_length
		self.vocabulary_size = vocabulary_size
		self.embedding_lookup = embedding_lookup
		self.scope = scope
		self.add_cells()

	def add_cells(self):
		"""
		Builds the graph to create the RNN's which do the dirty work
		===================================
		"""
		with tf.variable_scope(self.scope):
			cells = [tf.contrib.BasicRNNCell(self.hidden_dimension), tf.contrib.BasicLSTMCell(self.hidden_dimension)]
			self.cell = tf.contrib.rnn.MultiRNNCell(cells)
			self.vocab_logits_layer = Dense(self.vocabulary_size, activation=None)

	def generate_question(self, states, true_questions=None, true_question_lengths=None, supervised_training=False):
		"""
		Builds the graph to take in the state, and generate a question
		===================================
		INPUTS:
		states: float of shape (batch_size, hidden_dimension) - The state/history encoding for this round of dialog
		true_questions: float of shape (batch_size, max_question_length, embedding_size) || Assumed that questions have been padded to max_question_size
		true_question_lengths: int of shape(batch_size) - How long is the actual question?
		supervised_training: bool True: supervised pretraining|| False: RL training
		===================================
		OUTPUTS:
		final_outputs: float of shape (batch_size, max_sequence_length, vocabulary_size): The sequence of output vectors for every timestep
		final_sequence_lengths = (batch_size): The actual length of the questions
		"""
		with tf.variable_scope(self.scope):
			if supervised_training:
				helper = tf.contrib.seq2seq.TrainingHelper(true_questions, true_question_lengths, self.scope)
			else:
				start_tokens = tf.tile(self.start_token_embedding, [tf.shape(states)[0]])
				helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embedding_lookup, start_tokens=start_tokens, end_token=self.end_token_idx)
			decoder = tf.contrib.seq2seq.BasicDecoder(
				cell=self.cell,
				helper=helper,
				initial_states=states,
				output_layer=self.vocab_logits_layer,
			)
			#final sequence of outputs
			#final_outputs = (batch_size, max_sequence_length, hidden_size)
			#final_sequence_lengths = (batch_size)
			final_outputs, _, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, 
														impute_finished=True, maximum_iterations=self.max_question_length)
			return final_outputs, _, final_sequence_lengths
