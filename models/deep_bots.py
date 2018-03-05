import numpy as np
import tensorflow as tf

class ABot(object):
    """Abstracts an A-Bot for answering questions about a photo
    """
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

class QBot(object):
    """Abstracts a Q-Bot for asking questions about a photo
    """
    def encode_captions(self, captions):
        """Encodes captions.

        Args:
            captions: [Batch Size, Caption Encoding Dimensions]
        Returns:
            captions: encoded captions  
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def encode_facts(self, questions, answers):
        """Encodes questions and answers into a fact (Fact Encoder)

        Args:
            questions: questions asked by the Q-Bot in the most recent round [Batch Size, 1]
            answers: answers given by the A-Bot in response to the questions [Batch Size, 1]
        Returns:
            facts: encoded facts that combine the question and answer
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

    def encode_state_histories(self, prev_states, recent_facts):
        """Encodes states as a combination of facts for a given round (State/History Encoder)

        Args:
            prev_states: [(q_1, a_1, q_2, a_2) ... ] List of tuples
            recent_facts: [(q_n, a_n), ...] List of yuples
        Returns:
            state: encoded state that combines the current facts and previous facts [Batch Size, 1]
        """
        raise NotImplementedError("Each Q-Bot must re-implement this method.")

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

    def get_q_values(self, states):
        """Returns all Q-values for all actions given states

        Args:
            state: encoded states [Batch Size, 1]
        Returns:
            values: mapping of actions to expected return values [Batch Size, 1]
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