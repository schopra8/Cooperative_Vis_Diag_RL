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
		self.embedding_matrix = tf.get_variable("word_embeddings", shape=[self.config.VOCAB_SIZE, self.config.EMBEDDING_SIZE])

	def add_placeholders(self):
		pass

	def run_dialog(self, images, captions, true_questions, true_question_lengths, true_answers, true_answer_lengths, num_dialog_rounds=10, supervised_learning_rounds = 10):
		"""
		Function that runs dialog between two bots for a variable number of rounds, with a subset of rounds with supervised training
		================================
		INPUTS:
		images: float The embedded (VGG-16)images for this batch of dialogs: shape (batch_size, image_embedding_size)
		captions: int The captions as indices for this batch of dialogs: shape (batch_size, max_caption_length) (PADDED)
		true_questions : int The true questions for the entire dialog: (batch_size, num_dialog_rounds, max_question_length) (Not embedded)
		true_question_lengths : int The lengths of the true questions for the entire dialog (batch_size, num_dialog_rounds)
		true_answers : int The true answers for the entire dialog: (batch_size, num_dialog_rounds, max_answer_length) (Not embedded)
		true_answer_lengths :int The lengths of the true answers for the entire dialog (batch_size, num_dialog_rounds)
		num_dialog_rounds: int Number of rounds to run this dialog for
		supervised_learning_rounds: int The number of supervised learning rounds
		================================
		OUTPUTS:
		loss: float loss for the entire batch
		"""
		#Embedding lookup - padding (Add lengths)
		#dialog has questions and answers as indices. Not embedded. But padded.
		captions = self.embedding_lookup(captions)
		Q_state = self.Qbot.encode_captions(captions, caption_lengths)
		A_state = self.Abot.encode_images_captions(images, captions, caption_lengths)
		A_fact = self.Abot.encode_facts(captions, caption_lengths)
		prev_image_guess = self.Qbot.generate_image_representations(Q_state)
		loss = 0
		for i in xrange(num_dialog_rounds):
			if i >= supervised_learning_rounds: ## RL training
				#Q-Bot generates question logits
				question_logits, question_lengths =  self.Qbot.get_questions(Q_state, supervised_training = False)
				#Find embeddings of questions
				questions = self.embedding_lookup(tf.argmax(question_logits, axis = 2))
				#A-bot encodes questions
				encoded_questions = self.Abot.encode_questions(questions)
				#A-bot updates state
				A_state = self.Abot.encode_state_histories(images, captions, encoded_questions, A_fact, A_state)
				#Abot generates answer logits
				answer_logits, answer_lengths = self.Abot.get_answers(A_state, supervised_training = False)
				#Generate facts for that round of dialog
				facts, fact_lengths = self.concatenate_q_a(question_logits, question_lengths, answer_logits, answer_lengths)
				#Embed the facts into word vector space
				facts = self.embedding_lookup(facts)
				#Encode facts into both bots
				A_fact = self.Abot.encode_facts(facts, fact_lengths)
				Q_fact = self.Qbot.encode_facts(facts, fact_lengths)
				Q_state = self.Qbot.encode_state_histories(Q_state, facts, fact_lengths)
				# QBot Generates guess of image
				image_guess = self.Qbot.generate_image_representations(Q_state)
				
				#Calculate loss for this round
				reward = tf.reduce_sum(tf.square(prev_image_guess - images), axis = 1) - tf.reduce_sum(tf.square(image_guess - images), axis = 1)
				prev_image_guess = image_guess
				#### CHANGE HERE FOR UPDATING ONLY SINGLE BOT
				prob_questions = tf.argmax(tf.nn.softmax(question_logits, axis = 2), axis = 2)
				prob_answers = tf.argmax(tf.nn.softmax(answer_logits, axis = 2), axis = 2)
				negative_log_prob_exchange = -tf.log(prob_questions)-tf.log(prob_answers)
				loss += tf.reduce_mean(negative_log_prob_exhchange*rewards)

			else: ## Supervised training
				## ACCESS TRUE QUESTIONS  AND ANSWERS FOR THIS ROUND OF DIALOG
				questions = true_questions[:,i,:]
				answers = true_answers[:,i,:]
				true_question_lengths_round = true_question_lengths[:,i]
				true_answer_lengths_round = true_answer_lengths[:,i]
				#Generate questions based on current state
				question_logits, question_lengths =  self.Qbot.get_questions(Q_state, supervised_training = True)
				#Encode the true questions
				encoded_questions = self.Abot.encode_questions(self.embedding_lookup(questions))
				#Update A state based on true question
				A_state = self.Abot.encode_state_histories(images, captions, encoded_questions, A_fact, A_state)
				# ABot Generates answers based on current state
				answer_logits, answer_lengths = self.Abot.get_answers(A_state, supervised_training = True)
				#Generate facts from true questions and answers
				facts, fact_lengths = self.concatenate_q_a(questions, true_question_lengths_round, answers, true_answer_lengths_round)
				facts = self.embedding_lookup(facts)
				#Update state histories using current facts
				A_fact = self.Abot.encode_facts(facts, fact_lengths)
				Q_fact = self.Qbot.encode_facts(facts, fact_lengths)
				Q_state = self.Qbot.encode_state_histories(Q_state, facts, fact_lengths)
				#Guess image
				image_guess = self.Qbot.generate_image_representations(Q_state)
				#### Loss for supervised training
				dialog_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = question_logits, labels = questions)
				dialog_loss += tf.nn.sparse_softmax_cross_entropy_with_logits(logits = answer_logits, labels = answers)
				image_loss += tf.nn.l2_loss(image_guess - images)
				loss += dialog_loss + image_loss

			return loss


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

