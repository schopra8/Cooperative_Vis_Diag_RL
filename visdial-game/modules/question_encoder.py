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
        with tf.variable_scope(self.scope):
            self.cell = tf.contrib.rnn.GRUCell(self.hidden_dimension)

    def encode_questions(self, questions, question_lengths):
        """
        Given a question, output an embedding for the question.
        ===================================
        INPUTS:
        questions: float - (batch_size, max_sequence_length)
        """
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(questions)[0]
            _, encoded_questions = tf.nn.dynamic_rnn(
                self.cell,
                questions,
                dtype=tf.float32, sequence_length = question_lengths
            )
            return encoded_questions
