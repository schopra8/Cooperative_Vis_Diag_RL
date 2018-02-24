import tensorflow as tf
import numpy as np
from bots import SyntheticQBot
from bots import SyntheticABot

class Dialog_Bots(object):
	def __init__(self, config):
		""" Sets up the configuration parameters, creates Q Bot + A Bot. 
		"""
		self.config=config
		self.Qbot = SyntheticQBot(self.config.Q)
		self.Abot = SyntheticABot(self.config.A)
		self.sess = tf.Session()

	def add_placeholders(self):
		""" Construct placeholders for images and captions, for TF Graph.
        """
		self.image_placeholder = tf.placeholder(dtype = tf.int32, shape = [None, self.config.input_dimension])
		self.caption_placeholder = tf.placeholder(dtype = tf.int32, shape = [None, self.config.caption_dimension])


	def create_feed_dict(self, inputs, captions):
		""" Construct feed dict for images and camptions.
		"""
		feed_dict = {
			self.image_placeholder : inputs,
			self.caption_placeholder : captions
		}
		return feed_dict

	def run_dialog(self,batch_size=self.config.batch_size, rounds_dialog = 2, synthetic=True):
		""" Runs dialog for specified number of rounds:
				1) Q Bot asks question
				2) A Bot answers question based on history 
				3) A Bot encodes question for later usage in it's history
				4) Q bot encodes the answer and updates state history
        Args:
            rounds_dialog (int): Number of times this must repeat
        Returns:
            answer: The predictions of the Q bot at the end of (every) dialog
        """
		#First encode the caption and image in both bots
		q_bot_states = self.Qbot.encode_captions(self.caption_placeholder)
		a_bot_states  = self.Abot.encode_captions_images(self.caption_placeholder, self.image_placeholder)
		trajectories = [] * batch_size
		if synthetic:
			a_bot_recent_facts = [(-1, -1)] * batch_size # Sentinels for A Bot Fact 0
			q_bot_facts = [] 
		else:
			continue # TODO: Not Yet Implemented
		guesses = []
		for _ in xrange(rounds_dialog):
			questions = self.Qbot.get_questions(q_bot_states) # QBot generates questions (Q_t)
			question_encodings = self.Abot.encode_questions(questions) # ABot encodes questions (Q_t)
			a_bot_states = self.Abot.encode_state_histories(	# ABot encodes states (State, Y, C, Q_t, F_{t-1})
				self.image_placeholder,
				self.caption_placeholder,
				question_encodings,
				a_bot_recent_facts,
				a_bot_states
			)
			answers = self.Abot.decode_answers(a_bot_states) # ABot generates answers (A_t)

			a_bot_recent_facts = self.Abot.encode_facts(question_encodings, answers) # ABot generates facts (F_t)
			q_bot_facts = self.Qbot.encode_facts(questions, answers) # QBot encodes facts (F_t)
			q_bot_states = self.Qbot.encode_state_histories(q_bot_states, q_bot_facts) # QBot encode states
			if self.config.guess_every_round:
				guesses.append(self.Qbot.generate_image_representations(q_bot_states))
			for i, q in enumerate(questions):
				trajectories[i].append(q)
				trajectories[i].append(answers[i])
		if not self.config.guess_every_round:
			guesses = self.Qbot.generate_image_representations(q_bot_states)

		# TODO: Rewrite trajectories + after confirmation from others.
		# There should actually be two sets of trajectories. One set for Q Bot and another for A Bot.
		# The Trajectories should be in the form of [(state, action), ...]
		# Note: They are currently of the form [(q, a), ...], which I believe is incorrect (Sahil)
		return trajectories, guesses, q_bot_states, a_bot_states

	def get_minibatches(self, batch_size=20):
		# TODO Implement batching of captions, images
		pass

	def get_returns(self, trajectories, guesses, answers, gamma):
		""" Gets returns for a list of trajectories. 
			+1 Reward if guess == answer
			+0 Otherwise
		"""
		# TODO: Confirm the reward structure
		all_returns = []
		for i, g in enumerate(guesses):
			path_returns = [0] * len(trajectories[i])
			if g == answers[i]:
				# Correct answer, otherwise vector of all 0s
				discount_factors = np.power(gamma, np.arange(len(trajectories[i])))
				path_returns = list(np.multiply(np.ones(len(trajectories[i]), discount_factors)))
			all_returns.append(path_returns)
		return all_returns

	def train(self, batch_size=20, num_iterations=500, max_dialog_rounds=1):
		for i in num_iterations:
			inputs, captions, answers = self.get_minibatches(batch_size)
			guess_op = self.run_dialog(batch_size, max_dialog_rounds)
			feed = self.create_feed_dict(inputs, captions)
			trajectories, guesses, q_bot_states, a_bot_states = self.sess.run(guess_op, feed_dict = feed)

			# TODO: Perform Q Bot and A Bot Updates
			# Pseudocode:
			# 	Get Returns for trajectory
			#	Iterate over each of the trajectories, examinin the state, action -> reward pairings
			#	Mantain a default dict of (state, action) -> np.array([a_1, a_2, a_3])
			# 	Update entries in this numpy array
			#	Eventually perform running average between these numpy arrays and those stored in the appropriate
			#	Q Tables for Q Bot or A Bot
			if i % 2 == 0:
				# Update Q Bot
				returns = self.get_returns(trajectories, guesses, answers, self.config.Q.gamma)
			else:
				# Update A Bot
				returns = self.get_returns(trajectories, guesses, answers, self.config.A.gamma)





		 	
