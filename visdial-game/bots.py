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

class DeepABot():
    """Abstracts an A-Bot for answering questions about a photo
    """
    def __init__(self, config, start_token_embedding, embedding_lookup):
        with tf.variable_scope("a_bot") as scope:
            self.config = config
            self.fact_encoder = FactEncoder(self.config.hidden_dims, scope)
            self.question_encoder = QuestionEncoder(
                self.config.hidden_dims,
                scope
            )
            self.answer_decoder = AnswerDecoder(
                hidden_dimension=self.config.hidden_dims,
                start_token_embedding=start_token_embedding,
                end_token_idx=self.config.END_TOKEN_IDX,
                max_answer_length=self.config.MAX_ANSWER_LENGTH,
                vocabulary_size=self.config.VOCAB_SIZE,
                embedding_lookup=embedding_lookup
                scope
            )
            self.history_encoder = AHistoryEncoder(
                self.config.hidden_dims,
                scope
            )
    def encode_images_captions(self, captions, images, caption_lengths):
        """Encodes the captions and the images into initial states (S0) for A Bot.

        Args:
            captions [Batch Size, Caption] : Gives the captions for the current round of dialog
            images [Batch Size, Image] : The images for the dialog
            caption_lengths [Batch Size]: Gives lengths of the captions
        """
        batch_size = tf.shape(captions)[0]
        empty_questions = tf.zeros([batch_size, self.config.MAX_QUESTION_LENGTH])
        encoded_captions = self.fact_encoder.generate_fact(captions, caption_lengths)
        _, initial_states = self.history_encoder.generate_next_state(encoded_captions, empty_questions, images, prev_states=None)
        return initial_states

    def encode_questions(self, questions):
        """Encodes questions (Question Encoder)

        Args:
            questions: questions that the Q-Bot asked [Batch Size, max_question_length, vocab_size]
        Returns:
            question_encodings: encoding of the questions [Batch Size, hidden_size]
        """
        return self.question_encoder.encode_questions(questions)

    def encode_facts(self, inputs, input_lengths):
        """Encodes questions and answers into a fact (Fact Encoder)

        Args:
            inputs: concatenation of questions and answers [Batch Size, max_question_length + max_answer_length, vocab_size]
            input_lengths: true lengths of each q,a concatenation
        Returns:
            facts: encoded facts that combine the question and answer
        """
        return self.fact_encoder.generate_fact(inputs, input_lengths)

    def encode_state_histories(self, recent_facts, images, question_encodings, prev_states):
        """Encodes states as a combination of facts (State/History Encoder)

        Args:
            images [Batch Size, VGG Image Representation] : The images for the dialog
            question_encodings: encoded questions [Batch Size, hidden size]
            recent_facts: encoded facts [Batch Size, hidden size]
            prev_states: prev states [Batch Size, hidden size]
        Returns:
            states: encoded states that combine images, captions, question_encodings, recent_facts
        """
        states = self.history_encoder.generate_next_state(recent_facts, question_encodings, images, prev_states)
        return states

    def get_answers(self, states, true_answers=None, true_answer_lengths=None, supervised_training=False):
        """Returns answers according to some exploration policy given encoded states

        Args:
            state: encoded states [Batch Size, hidden dim]
        Returns:
            answers: answers that A Bot will provide [Batch Size, max_answer_length, vocabulary_size]
            answer_lengths: lengths of individaul answers [Batch Size] (since answres are zero padded )
        """
        if supervised_training:
            assert (true_answers != None and true_answer_lengths != None)
        answer_logits, answer_lengths = self.answer_decoder.generate_answer(
            states,
            true_answers,
            true_answer_lengths,
            supervised_training
        )
        return answer_logits, answer_lengths

class DeepQBot():
    """Abstracts a Q-Bot for asking questions about a photo
    """
    def __init__(self, config, start_token_embedding, embedding_lookup):
        with tf.variable_scope("q_bot") as scope:
            self.config = config
            self.fact_encoder = FactEncoder(self.config.hidden_dims, scope)
            self.question_decoder = QuestionDecoder(
                hidden_dimension=self.config.hidden_dims,
                start_token_embedding=start_token_embedding,
                end_token_idx=self.config.END_TOKEN_IDX,
                max_question_length=self.config.MAX_QUESTION_LENGTH,
                vocabulary_size=self.config.VOCAB_SIZE,
                embedding_lookup=embedding_lookup,
                scope
            )
            self.history_encoder = QHistoryEncoder(
                self.config.hidden_dims,
                scope
            )
            self.feature_regressor = FeatureRegression(self.config.IMG_REP_DIM, scope)

    def encode_captions(self, captions, caption_lengths):
        """Encodes captions.

        Args:
            captions: float of shape (batch size, max_caption_length, embedding_dims)
        Returns:
            captions: encoded captions  
        """
        fact_captions = self.fact_encoder.generate_next_fact(captions, caption_lengths)
        state_captions = self.history_encoder.generate_next_state(fact_captions)
        return state_captions

    def encode_facts(self, inputs, input_lengths):
        """Encodes questions and answers into a fact (Fact Encoder)

        Args:
            inputs: concatenation of questions and answers [Batch Size, max_question_length + max_answer_length, vocab_size]
            input_lengths: true lengths of each q,a concatenation
        Returns:
            facts: encoded facts that combine the question and answer
        """
        return self.fact_encoder.generate_next_fact(inputs, input_lengths)

    def encode_state_histories(self, recent_facts, prev_states=None):
        """Encodes states as a combination of facts for a given round (State/History Encoder)

        Args:
            prev_states
            recent_facts
        Returns:
            state: encoded state that combines the current facts and previous facts [Batch Size, 1]
        """
        outputs, next_states = self.history_encoder.generate_next_state(recent_facts, prev_states)
        return outputs, next_states

    def generate_image_representations(self, states):
        """Guesses images given the current states (Feature Regression Network)

        Args:
            state: encoded states [Batch Size, hiden_size]
        Returns:
            image_repr: representation of the predicted images [Batch Size, IMG_REP_DIM]
        """
        return self.feature_regressor.generate_image_prediction(states)

    def get_questions(self, states, true_questions=None, true_question_lengths=None, supervised_training=False):
        """Returns questions according to some exploration policy given encoded states

        Args:
            state: encoded states [Batch Size, hidden size]
        Returns:
            questions: questions that Q Bot will ask [Batch Size, max_question_length]
        """
        if supervised_training:
            assert (true_questions != None and true_question_lengths != None)
        question_logits, question_lengths = self.question_decoder.generate_question(
            states,
            true_questions=true_questions,
            true_question_lengths=true_question_lengths,
            supervised_training=supervised_training
        )
        return question_logits, question_lengths
