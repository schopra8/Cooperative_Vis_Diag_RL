import tensorflow as tf
import numpy as np
from bots import DeepQBot
from bots import DeepABot

class model():
	def __init__(self, config):
		""" Sets up the configuration parameters, creates Q Bot + A Bot.
		"""
		self.config= config
		self.Qbot = DeepQBot()
		self.Abot = DeepABot()

	def run_dialog(self, images, captions, dialog, dialog_lengths, num_dialog_rounds=10, curriculum_learning_rounds = 10):
		Q_state = self.Qbot.encode_captions(captions, caption_lengths)
		A_state = self.Abot.encode_images_captions(images, captions, caption_lengths)
		A_fact = self.Abot.encode_facts(captions, caption_lengths)
		prev_image_guess = self.Qbot.generate_image_representations(Q_state)
		loss = 0
		for i in xrange(num_dialog_rounds):
			if i >= curriculum_learning_rounds: ## RL training
				question_logits, question_lengths =  self.Qbot.get_questions(Q_state, supervised_training = False)
				questions = self.embedding_lookup(tf.argmax(question_logits, axis = 2))
				encoded_questions = self.Abot.encode_questions(questions)
				A_state = self.Abot.encode_state_histories(images, captions, encoded_questions, A_fact, A_state)
				answer_logits, answer_lengths = self.Abot.get_answers(A_state)
				facts, fact_lengths = self.concatenate_q_a(question_logits, question_lengths, answer_logits, answer_lengths)
				facts = self.embedding_lookup(facts)
				A_fact = self.Abot.encode_facts(facts, fact_lengths)
				Q_fact = self.Qbot.encode_facts(facts, fact_lengths)
				Q_state = self.Qbot.encode_state_histories(Q_state, facts, fact_lengths)
				image_guess = self.Qbot.generate_image_representations(Q_state)
				reward = tf.reduce_sum(tf.square(prev_image_guess - images), axis = 1) - tf.reduce_sum(tf.square(image_guess - images), axis = 1)
				#### CHANGE HERE FOR UPDATING ONLY SINGLE BOT
				prob_questions = tf.argmax(tf.nn.softmax(question_logits, axis = 2), axis = 2)
				prob_answers = tf.argmax(tf.nn.softmax(answer_logits, axis = 2), axis = 2)
				negative_log_prob_exchange = -tf.log(prob_questions)-tf.log(prob_answers)
				loss += tf.reduce_mean(negative_log_prob_exhchange*rewards)

			else: ## Supervised training
				question_logits, question_lengths =  self.Qbot.get_questions(Q_state, supervised_training = True)
				questions = dialog.questions ## ACCESS TRUE QUESTIONS FOR THIS ROUND OF DIALOG FROM Q_BOT
				encoded_questions = self.Abot.encode_questions(questions)
				A_state = self.Abot.encode_state_histories(images, captions, encoded_questions, A_fact, A_state)
				answer_logits, answer_lengths = self.Abot.get_answers(A_state)
				facts, fact_lengths = self.concatenate_q_a(question_logits, question_lengths, answer_logits, answer_lengths)
				facts = self.embedding_lookup(facts)
				A_fact = self.Abot.encode_facts(facts, fact_lengths)
				Q_fact = self.Qbot.encode_facts(facts, fact_lengths)
				Q_state = self.Qbot.encode_state_histories(Q_state, facts, fact_lengths)
				image_guess = self.Qbot.generate_image_representations(Q_state)
				#### CHANGE HERE FOR UPDATING ONLY SINGLE BOT
				dialog_loss = tf.nn.sparse_cross_entropyquestion_logits
				loss += tf.reduce_mean(negative_log_prob_exhchange*rewards)

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

