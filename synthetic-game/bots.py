import os, sys
sys.path.append('../')

import numpy as np

from collections import defaultdict

from models.qbot import QBot
from models.abot import ABot


class SyntheticQBot(QBot):
    def __init__(self, config):
        self.state = ()
        self.Q = defaultdict(lambda: np.zeros(config.num_actions))
        self.Q_regression = defaultdict(lambda: np.zeros(config.num_classes))

        # {epsilon, num_actions, num_classes}
        self.config = config
    
    def encode_caption(self, caption):
        """Encodes a question and answer into a fact (Fact Encoder)

        Args:
            caption (vector) : Gives the caption for the current round of dialog
        """
        self.state += caption
    
    def encode_fact(self, question, answer):
        """Encodes a question and answer into a fact (Fact Encoder)

        Args:
            question (int): A question asked by the Q-Bot in the most recent round
            answer (int): The answer given by the A-Bot in response to the question
        Returns:
            fact (tuple): An encoded fact that combines the question and answer
        """
        fact = (question, answer)
        self.fact = fact
        return fact

    def encode_state_history(self, fact = self.fact):
        """Encodes the state as a combination of facts (State/History Encoder)

        Args:
            fact (tuple): An encoded fact
        Returns:
            state (tuple): An encoded state that combines the current fact and previous facts
        """
        self.state += fact
        return self.state

    def decode_question(self, state):
        """Decodes (generates) a question given the current state (Question Decoder)

        Args:
            state: An encoded state
        Returns:
            question: A question that the Q-Bot will ask
        """
        question = np.argmax(self.Q[state])
        self.question = question
        return question

    def generate_image_representation(self, state):
        """Guesses an image given the current state (Feature Regression Network)

        Args:
            state: An encoded state
        Returns:
            image_repr: A representation of the predicted image
        """
        image_repr = np.argmax(self.Q_regression[state])
        return image_repr

    def get_q_values(self, state):
        """Returns all Q-values for all actions given state

        Args:
            state: An encoded state
        Returns:
            q_values: A representation of action to expected value
        """
        q_values = self.Q[state]
        return q_values

    def get_action(self, state):
        """Returns an action according to some exploration policy given an encoded state

        Args:
            state: An encoded state
        Returns:
            action: A question for the Q-Bot to ask
        """
        if np.random.rand() < self.config.epsilon:
            return np.random.choice(self.config.num_actions)
        else:
            return self.decode_answer(state)


class SyntheticABot(ABot):
    def __init__(self, config):
        self.state = ()
        self.Q = defaultdict(lambda: np.zeros(config.num_actions))
        self.fact = ()
        # {epsilon, num_actions}
        self.config = config

    def encode_caption_image(self, caption, image):
        """Encodes the caption and the image into the state

        Args:
            caption (vector) : Gives the caption for the current round of dialog
            image (vector) : The image for the dialog
        """
        self.state += image
        self.state += caption
    def encode_question(self, question):
        """Encodes a question given the current state (Question Encoder)

        Args:
            question: A question that the Q-Bot asked
        Returns:
            question_encoding: An encoding of the question
        """
        question_encoding = question
        self.question_encoding = question
        return question_encoding

    def encode_fact(self, question, answer):
        """Encodes a fact as a combination of a question and answer (Fact Encoder)

        Args:
            question: A question asked by the Q-Bot in the most recent round
            answer: The answer given by the A-Bot in response to the question
        Returns:
            fact: An encoded fact that combines the question and answer
        """
        fact = (question, answer)
        self.fact = fact
        return fact

    def encode_state_history(self, question_encoding, fact = self.fact):
        """Encodes a state as a combination of facts (State/History Encoder)

        Args:
            question_encoding: An encoded question
            fact: An encoded fact
        Returns:
            state: An encoded state that combines question_encodings and facts
        """
        self.state += (question_encoding,) + fact
        return self.state

    def decode_answer(self, state):
        """Generates an answer given the current state (Answer Decoder)

        Args:
            state: An encoded state
        Returns:
            answer: An answer
        """
        answer = np.argmax(self.Q[state])
        return answer

    def get_q_values(self, state):
        """Returns all Q-values for all actions given state

        Args:
            state: An encoded state
        Returns:
            q_values: A representation of action to expected value
        """
        q_values = self.Q[state]
        return q_values

    def get_action(self, state):
        """Returns an action according to some exploration policy given an encoded state

        Args:
            state: An encoded state
        Returns:
            action: A question for the Q-Bot to ask
        """
        if np.random.rand() < self.config.epsilon:
            return np.random.choice(self.config.num_actions)
        else:
            return self.decode_answer(state)


if __name__ == '__main__':
    q_config = {
        'epsilon': 0.6,
        'num_actions': 3,
        'num_classes': 144
    }
    qbot = SyntheticQBot(q_config)
    a_config = {
        'epsilon': 0.6,
        'num_actions': 4
    }
    abot = SyntheticABot(a_config)
