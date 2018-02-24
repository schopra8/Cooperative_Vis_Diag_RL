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

	def run_dialog(self, rounds_dialog = 2, synthetic=True):
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
		if synthetic:
			a_bot_recent_facts = [(-1, -1)] * self.config.batch_size # Sentinels for A Bot Fact 0
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
		#Final guess if not already added
		if not self.config.guess_every_round:
			guesses = self.Qbot.generate_image_representations(q_bot_states)
		return guesses

	def add_prediction_op(self, inputs, captions):
		guess = self.run_dialog(self.config.max_dialog_rounds)
		feed = self.create_feed_dict(inputs, captions)
		guesses = self.sess.run(guess, feed_dict = feed)
	
