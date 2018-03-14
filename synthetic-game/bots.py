import os, sys
import numpy as np
from collections import defaultdict

class SyntheticQBot(object):
    def __init__(self, config):
        self.config = config
        self.Q = defaultdict(lambda: np.zeros(self.config.num_actions))
        self.Q_regression = defaultdict(lambda: np.zeros(self.config.num_classes))
        self.states = None

    def set_initial_states(self, game_types):
        """
            Reset bot states and initialize to the game types.
        """
        self.states = [((game_type,),) for game_type in game_types] 

    def decode_questions(self, epsilon):
        """
            Generate question. Optimal question selected with epsilon probability.
        """
        def gen_question(state, epsilon):
            optimal_question = np.argmax(self.Q[state])

            if np.random.rand() <= epsilon:
                return optimal_question
            else:
                print 'wtf bro'
                suboptimal_question = np.random.choice(self.config.num_actions)
                while suboptimal_question == optimal_question:
                    suboptimal_question = np.random.choice(self.config.num_actions)
                return suboptimal_question
        return [gen_question(state, epsilon) for state in self.states]

    def update_states(self, questions, answers):
        """
            Update state, with question answer tuple.
        """
        for i in xrange(len(questions)):
            q = questions[i]
            a = answers[i]
            self.states[i] += ((q, a),)

    def generate_image_predictions(self, epsilon):
        """
            Guesses an image given the current state (Feature Regression Network)
        """
        def gen_image_prediction(state, epsilon):
            optimal_prediction = np.argmax(self.Q_regression[state])

            if np.random.rand() <= epsilon:
                return optimal_prediction
            else:
                print 'wtf bro'
                suboptimal_prediction = np.random.choice(self.config.num_classes)
                while suboptimal_prediction == optimal_prediction:
                    suboptimal_prediction = np.random.choice(self.config.num_classes)
                return suboptimal_prediction
        return [gen_image_prediction(state, epsilon) for state in self.states]

class SyntheticABot(object):
    def __init__(self, config):
        self.config = config
        self.Q = defaultdict(lambda: np.zeros(self.config.num_actions))
        self.states = None

    def set_initial_states(self, game_types, images, questions):
        """
            Reset bot state and initialize to the game type.
        """
        self.states = []
        for i in xrange(len(game_types)):
            game_type = game_types[i]
            image = images[i]
            question = questions[i]
            self.states.append(((game_type, image, question),))

    def decode_answers(self, epsilon):
        """
            Generate answers. Optimal answer selected with epsilon probability.
        """
        def gen_answer(state, epsilon):
            optimal_answer = np.argmax(self.Q[state])

            # Don't just default to first indexed element, if all value are 0
            if self.Q[state][optimal_answer] == 0:
                optimal_answer = np.random.choice(self.config.num_actions)

            if np.random.rand() <= epsilon:
                return optimal_answer
            else:
                print 'wtf bro'
                suboptimal_answer = np.random.choice(self.config.num_actions)
                while suboptimal_answer == optimal_answer:
                    suboptimal_answer = np.random.choice(self.config.num_actions)
                return suboptimal_answer
        return [gen_answer(state, epsilon) for state in self.states]

    def update_states(self, prev_answers, questions):
        """
            Update state, with prev_answers, questions tuple.
        """
        for i in xrange(len(prev_answers)):
            prev_answer = prev_answers[i]
            question = questions[i]
            self.states[i] += ((prev_answer, question),)
