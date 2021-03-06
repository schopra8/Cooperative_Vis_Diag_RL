import tensorflow as tf
from tensorflow.python.layers.core import Dense

class AnswerDecoder(object):
    """
        Takes in previous state, and returns the answer for that time step
        ### Dimensions
        Previous state: (batch_size, hidden_dimension)
        answers: (batch_size, answer_length, indices)
    """
    def __init__(self, hidden_dimension, start_token_idx, end_token_idx, max_answer_length, vocabulary_size, embedding_matrix, scope):
        """
        Initialization function
        ====================
        INPUTS:
        hidden_dimension: int - shape of the hidden state for the LSTM/RNN cells used
        start_token_idx : 
        end_token_idx: int of index referring to the end token within our vocabulary
        max_answer_length : int - length of longest answers
        vocabulary_size: int - size of vocabulary (including start and end tokens)
        embedding_matrix: embedding_matrix
        scope: variable scope of decoder
        ====================
        """
        self.hidden_dimension = hidden_dimension
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.max_answer_length = max_answer_length
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
            # Stacked Cells: Inputs -> Basic RNN Cell -> Basic LSTM Cell -> Outputs
            # cells = [tf.contrib.rnn.GRUCell(self.hidden_dimension)]
            # self.cell = tf.contrib.rnn.MultiRNNCell(cells)
            self.cell = tf.contrib.rnn.GRUCell(self.hidden_dimension)
            self.vocab_logits_layer = Dense(self.vocabulary_size, activation=None)

    def embedding_lookup(self, indices):
        return tf.nn.embedding_lookup(self.embedding_matrix, indices)

    def generate_answer(self, states, true_answers, true_answer_lengths, supervised_training=True):
        """
        Builds the graph to take in the state, and generate an answer
        ===================================
        INPUTS:
        states: float of shape (batch_size, hidden_dimension) - The state/history encoding for this round of dialog
        true_answers: float of shape (batch_size, max_answer_length, embedding_size) || Assumed that answers have been padded to max_answer_length
        true_answer_lengths: int of shape(batch_size) - How long is the actual answers?
        supervised_training: bool True: supervised pretraining || False: RL training
        ===================================
        OUTPUTS:
        final_outputs: float of shape (batch_size, max_sequence_length, hidden_size): The sequence of output vectors for every timestep
        final_sequence_lengths = (batch_size): The actual length of the answers
        """
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            if supervised_training:
                embedded_answers = self.embedding_lookup(true_answers)
                helper = tf.contrib.seq2seq.TrainingHelper(embedded_answers, true_answer_lengths, time_major = False)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.cell,
                helper=helper,
                initial_state=states,
                output_layer=self.vocab_logits_layer,
                )
            # final_outputs = (batch_size, max_sequence_length, hidden_size)
            # final_sequence_lengths = (batch_size)
                final_outputs, _, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, 
                                                        impute_finished=False)
                return final_outputs.rnn_output, final_sequence_lengths, final_outputs.sample_id
            else:
                start_tokens = tf.ones([tf.shape(states)[0]], dtype=tf.int32) * self.start_token_idx
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embedding_lookup, start_tokens=start_tokens, end_token=self.end_token_idx)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=self.cell,
                    helper=helper,
                    initial_state=states,
                    output_layer=self.vocab_logits_layer,
                )
                final_outputs, _, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, 
                                                        impute_finished=False, maximum_iterations=self.max_answer_length)
                return final_outputs.rnn_output, final_sequence_lengths, final_outputs.sample_id