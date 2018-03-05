import tensorflow as tf
import os, sys
from modules.a_history_encoder import AHistoryEncoder
from modules.answer_decoder import AnswerDecoder
from modules.fact_encoder import FactEncoder
from modules.feature_regression import FeatureRegression
from modules.q_history_encoder import QHistoryEncoder
from modules.question_decoder import QuestionDecoder
from modules.question_encoder import QuestionEncoder

sys.path.append('../')
from models.qbot import QBot
from models.abot import ABot

class DeepABot(ABot):
    """Abstracts an A-Bot for answering questions about a photo
    """
    def __init__(self, config):
        with tf.variable_scope("a_bot") as scope:
            self.config = config
            self.fact_encoder = FactEncoder(self.config.hidden_dims, scope)
            self.question_encoder = QuestionDecoder(
                self.config.hidden_dims,
                scope
            )
            self.answer_decoder = AnswerDecoder(
                self.config.hidden_dims,
                self.config.START_TOKEN,
                self.config.END_TOKEN,
                self.config.MAX_ANSWER_LENGTH,
                scope
            )
            self.history_encoder = QHistoryEncoder(
                self.config.hidden_dims,
                scope
            )        

    def encode_images_captions(self, captions, images):
        """Encodes the captions and the images into the states

        Args:
            captions [Batch Size, Caption] : Gives the captions for the current round of dialog
            images [Batch Size, Image] : The images for the dialog
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")

    def encode_questions(self, questions):
        """Encodes questions (Question Encoder)

        Args:
            questions: questions that the Q-Bot asked [Batch Size, 1]
        Returns:
            question_encodings: encoding of the questions [Batch Size,]
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")

    def encode_facts(self, questions, answers):
        """Encodes questions and answers into a fact (Fact Encoder)

        Args:
            questions: questions asked by the Q-Bot in the most recent round [Batch Size]
            answers: answers given by the A-Bot in response to the questions [Batch Size]
        Returns:
            facts: encoded facts that combine the question and answer
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def encode_state_histories(self, images, captions, question_encodings, recent_facts, prev_states):
        """Encodes states as a combination of facts (State/History Encoder)

        Args:
            images [Batch Size, Image] : The images for the dialog
            captions [Batch Size, Caption] : Gives the captions for the current round of dialog
            question_encodings: encoded questions [Batch Size]
            recent_facts: encoded facts [Batch Size]
            prev_states: prev states [Batch Size]
        Returns:
            states: encoded states that combine images, captions, question_encodings, recent_facts
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")

    def decode_answers(self, states):
        """Generates answer given the current states (Answer Decoder)

        Args:
            states: encoded states
        Returns:
            answers: answers
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")

    def generate_image_representations(self, states):
        """Guesses images given the current states (Feature Regression Network)

        Args:
            state: encoded states [Batch Size, 1]
        Returns:
            image_repr: representation of the predicted images [Batch Size, 1]
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def get_q_values(self, states):
        """Returns all Q-values for all actions given states

        Args:
            state: encoded states [Batch Size, 1]
        Returns:
            values: mapping of actions to expected return values [Batch Size, 1]
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")

    def get_answers(self, states):
        """Returns answers according to some exploration policy given encoded states

        Args:
            state: encoded states [Batch Size, 1]
        Returns:
            answers: answers that A Bot will provide [Batch Size, 1]
        """
        raise NotImplementedError("Each A-Bot must re-implement this method.")

class DeepQBot(QBot):
    """Abstracts a Q-Bot for asking questions about a photo
    """
    def __init__(self, config):
        with tf.variable_scope("q_bot") as scope:
            self.config = config
            self.fact_encoder = FactEncoder(self.config.hidden_dims, scope)
            self.question_decoder = QuestionDecoder(
                self.config.hidden_dims,
                self.config.START_TOKEN,
                self.config.END_TOKEN,
                self.config.MAX_QUESTION_LENGTH,
                scope
            )
            self.history_encoder = QHistoryEncoder(
                self.config.hidden_dims,
                scope
            )

    def encode_captions(self, captions, caption_lengths):
        """Encodes captions.

        Args:
            captions: float of shape (batch size, max_caption_length, embedding_dims)
        Returns:
            captions: encoded captions  
        """
        fact_captions = self.fact_encoder.generate_fact_from_captions(captions, caption_lengths)
        state_captions = self.history_encoder.generate_next_state(fact_captions)
        return state_captions

    def encode_facts(self, questions, answers):
        """Encodes questions and answers into a fact (Fact Encoder)

        Args:
            questions: questions asked by the Q-Bot in the most recent round 
                       float of shape (batch size, max_question_length, embedding_dims)
            answers: answers given by the A-Bot in response to the questions 
                       float of shape (batch size, max_answer_length, embedding_dims)
        Returns:
            facts: encoded facts that combine the question and answer
        """
        return self.fact_encoder.generate_next_fact(questions, answers)

    def encode_state_histories(self, prev_states, recent_facts):
        """Encodes states as a combination of facts for a given round (State/History Encoder)

        Args:
            prev_states: [(q_1, a_1, q_2, a_2) ... ] List of tuples
            recent_facts: [(q_n, a_n), ...] List of yuples
        Returns:
            state: encoded state that combines the current facts and previous facts [Batch Size, 1]
        """
        return self.history_encoder.generate_next_state(recent_facts, prev_states)

    def decode_questions(self, states):
        """Decodes (generates) questions given the current states (Question Decoder)

        Args:
            states (list of tuples) [Batch Size]: encoded states
        Returns:
            questions: (list of ints) [Batch Size]
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def generate_image_representations(self, states):
        """Guesses images given the current states (Feature Regression Network)

        Args:
            state: encoded states [Batch Size, 1]
        Returns:
            image_repr: representation of the predicted images [Batch Size, 1]
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def get_questions(self, states):
        """Returns questions according to some exploration policy given encoded states

        Args:
            state: encoded states [Batch Size, 1]
        Returns:
            questions: questions that Q Bot will ask [Batch Size, 1]
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")