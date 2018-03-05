import tensorflow as tf
import numpy as np

class model():
	def __init__(self, config):
		""" Sets up the configuration parameters, creates Q Bot + A Bot.
		"""
		self.config= config

	def run_dialog(self, images, captions, num_dialog_rounds=2, test=False):
		pass

	def update_epsilon(self, iteration_num):
		pass

	def get_minibatches(self, batch_size=20):
		pass
			

	def get_returns(self, trajectories, predictions, labels, gamma):
		""" Gets returns for a list of trajectories.
			+1 Reward if guess == answer
			-1 Otherwise
		"""
		pass

	def train(self, batch_size=20, num_iterations=500, max_dialog_rounds=2):
	
		pass

	def evaluate(self, minibatch_generator, max_dialog_rounds):
		pass

	def show_dialog(self, image, caption, answer):
		pass

	def concatenate_q_a(self, questions, question_lengths, answers, answer_lengths):
		"""
		Concatenate question, answer pairs
		===================================
		INPUTS:
		questions: float of shape (batch_size, max_question_length) -- tensor where each row are indices into vocabulary
		question_lengths: int of shape (batch_size) -- tensor listing true length of each question in questions tensor
		answers: float of shape (batch_size, max_answer_length) -- tensor where each row are indices into vocabulary
		answer_lengths: int of shape (batch_size) -- tensor listing true length of each answer in answers tensor
		===================================
		OUTPUTS:
		question_answer_pairs: float of shape (batch_size, max_question_length + max_answer_length): The sequence of output vectors for every timestep
		question_answer_pair_lengths = (batch_size): The actual length of the question, answer concatenations
		"""
		batch_size = tf.shape(questions)[0]
		stripped_question_answer_pairs = [tf.concat([questions[i,0:question_lengths[i],:], answers[i,0:answer_lengths[i],:]], axis = 1)for i in xrange(batch_size)]
		max_size = self.config.MAX_QUESTION_LENGTH + self.config.MAX_ANSWER_LENGTH
		padded_question_answer_pairs = [tf.pad(stripped_question_answer_pairs[i], [0, max_size - tf.shape(stripped_question_answer_pairs[i])[0]]) for i in xrange(batch_size)]
		question_answer_pairs = tf.stack(padded_question_answer_pairs, axis = 0)
		question_answer_pair_lengths = tf.add(question_lengths, answer_lengths)
		return question_answer_pairs, question_answer_pair_lengths


	