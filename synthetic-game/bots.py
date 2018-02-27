import os, sys
sys.path.append('../')

import numpy as np

from collections import defaultdict

from models.qbot import QBot
from models.abot import ABot

class SyntheticQBot(QBot):
    def __init__(self, config):
        self.Q = defaultdict(lambda: np.zeros(config.num_actions))
        self.Q_regression = defaultdict(lambda: np.zeros(config.num_classes))

        # {epsilon, num_actions, num_classes}
        self.config = config

    def encode_captions(self, captions):
        """Encodes captions.

        Args:
            captions: [Batch Size, Caption Encoding Dimensions]
        Returns:
            captions: encoded captions
        """
        return [((c),) for c in captions] # [((c1)), ((c2))]

    def encode_facts(self, questions, answers):
        """Encodes questions and answers into facts (Fact Encoder)

        Args:
            questions [Batch Size]: List of ints, where int represents question asked
                                      by the Q-Bot in the most recent round
            answers [Batch Size]: List of ints, where int represents an answer by the A-Bot
                                      to the questions
        Returns:
            fact (tuple): An encoded fact that combines the question and answer
        """
        return zip(questions, answers)

    def encode_state_histories(self, prev_states, facts):
        """Encodes the state as a combination of facts (State/History Encoder)

        Args:
            prev_state (list of tuples) [Batch Size]
            facts (list of tuples) [Batch Size]
        Returns:
            state (list of tuples) [Batch Size]: states that combines the current fact and previous facts
        """
        new_states = [d_state + facts[i] for i, d_state in enumerate(prev_states)]
        return new_states

    def decode_questions(self, states):
        """Decodes (generates) questions given the current states (Question Decoder)

        Args:
            states (list of tuples) [Batch Size]: encoded states
        Returns:
            questions: (list of ints) [Batch Size]
        """
        questions = [np.argmax(self.Q[state]) for state in states]
        return questions

    def generate_image_predictions(self, states):
        """Guesses an image given the current state (Feature Regression Network)

        Args:
            states (list of tuples) [Batch Size]: encoded states
        Returns:
            image_representations [Batch Size, img_dim]: representation of the predicted images
        """
        predictions = [np.argmax(self.Q_regression[state]) for state in states]
        return predictions

    def get_q_values(self, states):
        """Returns all Q-values for all actions given state

        Args:
            states (list of tuples) [Batch Size]: encoded states
        Returns:
            q_values (list of maps) [Batch Size]: representations of actions to expected return values
        """
        q_values = [self.Q[state] for state in states]
        return q_values

    def get_questions(self, states):
        """Returns questions according to some exploration policy given encoded states

        Args:
            state: encoded states [Batch Size, 1]
        Returns:
            questions: questions that Q Bot will ask [Batch Size, 1]
        """
        optimal_questions = self.decode_questions(states)
        def get_question(optimal_question):
            if np.random.rand() < self.config.epsilon:
                return np.random.choice(self.config.num_actions)
            else:
                return optimal_question
        questions = [get_question(optimal_question) for optimal_question in optimal_questions]
        return questions

class SyntheticABot(ABot):
    def __init__(self, config):
        self.Q = defaultdict(lambda: np.zeros(config.num_actions))
        # {epsilon, num_actions}
        self.config = config

    def encode_images_captions(self, images, captions):
        """Encodes the images and the captions into the states

        Args:
            images [Batch Size, Image] : The images for the dialog
            captions [Batch Size, Caption] : Gives the captions for the current round of dialog
        Return:
            encoded_im_cap = [((Image, Caption)), ...]
        """
        encoded_im_cap = zip(images, captions)
        return [((im_cap),) for im_cap in encoded_im_cap]

    def encode_questions(self, questions):
        """Encodes questions (Question Encoder)

        Args:
            questions: questions that the Q-Bot asked [Batch Size, 1]
        Returns:
            question_encodings: encoding of the questions [Batch Size,]
        """
        return questions

    def encode_facts(self, questions, answers):
        """Encodes questions and answers into a fact (Fact Encoder)

        Args:
            questions: questions asked by the Q-Bot in the most recent round [Batch Size, 1]
            answers: answers given by the A-Bot in response to the questions [Batch Size, 1]
        Returns:
            facts: encoded facts that combine the images, captions, questions, and answers
        """
        facts = zip(questions, answers)
        return facts

    def encode_state_histories(self, images, question_encodings, recent_facts, prev_states):
        """Encodes states as a combination of facts (State/History Encoder)

        Args:
            question_encodings: encoded questions [Batch Size, 1]
            facts: encoded facts [Batch Size, 1]
        Returns:
            state: encoded states that combine question_encodings and facts
        """
        new_states = zip(images, question_encodings, recent_facts)
        histories = [state + (tuple(new_states[i]),) for i, state in enumerate(prev_states)]
        return histories

    def decode_answers(self, states):
        """Generates an answer given the current state s(Answer Decoder)

        Args:
            states: encoded states
        Returns:
            answer: answers
        """
        print states[0]
        answers = [np.argmax(self.Q[state]) for state in states]
        return answers

    def get_q_values(self, states):
        """Returns all Q-values for all actions given state

        Args:
            states: encoded states
        Returns:
            q_values: A representation of action to expected value
        """
        q_values = [self.Q[state] for state in states]
        return q_values

    def get_answers(self, states):
        """Returns answers according to some exploration policy given encoded states

        Args:
            state: encoded states [Batch Size, 1]
        Returns:
            answers: answers that A Bot will provide [Batch Size, 1]
        """
        optimal_answers = self.decode_answers(states)
        def get_answer(optimal_answer):
            if np.random.rand() < self.config.epsilon:
                return np.random.choice(self.config.num_actions)
            else:
                return optimal_answer
        answers = [get_answer(optimal_answer) for optimal_answer in optimal_answers]
        return answers


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
