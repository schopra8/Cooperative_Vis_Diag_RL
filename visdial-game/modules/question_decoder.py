import tensorflow as tf
from tensorflow.python.layers.core import Dense

class QuestionDecoder(object):
    """
        Takes in previous state, and returns the question for that time step
        ### Dimensions
        Previous state: (batch_size, hidden_dimension)
        Question: (batch_size, question_length, indices)
    """
    def __init__(self, hidden_dimension, start_token_idx, end_token_idx, max_question_length, vocabulary_size, embedding_matrix, scope):
        """
        Initialization function
        ====================
        INPUTS:
        hidden_dimension: int - shape of the hidden state for the LSTM/RNN cells used
        start_token_idx : 
        end_token_idx: int of index referring to the end token within our vocabulary
        max_question_length : int - length of longest question
        vocabulary_size: int - size of vocabulary (including start and end tokens)
        embedding_matrix: embedding matrix
        scope: variable scope of decoder
        ====================
        """
        self.hidden_dimension = hidden_dimension
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.max_question_length = max_question_length
        self.vocabulary_size = vocabulary_size
        self.embedding_matrix = embedding_matrix
        self.scope = scope
        self.add_cells()

    def add_cells(self):
        """
        Builds the graph to create the RNN's which do the dirty work
        ===================================
        """
        with tf.variable_scope(self.scope):
            # cells = [tf.contrib.rnn.GRUCell(self.hidden_dimension)]
            # self.cell = tf.contrib.rnn.MultiRNNCell(cells)
            self.cell = tf.contrib.rnn.GRUCell(self.hidden_dimension)
            self.vocab_logits_layer = Dense(self.vocabulary_size, activation=None)

    def embedding_lookup(self, indices):
        return tf.nn.embedding_lookup(self.embedding_matrix, indices)

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
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            if supervised_training:
                embedded_questions = self.embedding_lookup(true_questions)
                helper = tf.contrib.seq2seq.TrainingHelper(embedded_questions, true_question_lengths, time_major = False)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=self.cell,
                    helper=helper,
                    initial_state=states,
                    output_layer=self.vocab_logits_layer,
                )
                final_outputs, _ , final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, 
                                                        impute_finished=False)
                return final_outputs.rnn_output, final_sequence_lengths, final_outputs.sample_id
            else:
                start_tokens = tf.ones([tf.shape(states)[0]], dtype=tf.int32) * self.start_token_idx
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_matrix, start_tokens, self.end_token_idx)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=self.cell,
                    helper=helper,
                    initial_state=states,
                    output_layer=self.vocab_logits_layer,
                )
                final_outputs, _, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, 
                                                        impute_finished=False, maximum_iterations = self.max_question_length)
                return final_outputs.rnn_output, final_sequence_lengths, final_outputs.sample_id
