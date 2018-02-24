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

	def run_dialog(self, rounds_dialog = 2):
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
		qbot_state = self.Qbot.encode_captions(self.caption_placeholder)
		abot_state  = self.Abot.encode_captions_images(self.caption_placeholder, self.image_placeholder)
		guesses=[]
		trajectory=[]
		abot_fact= []
		for _ in xrange(rounds_dialog):
			#First question asked by Q bot
			question = self.Qbot.get_questions(qbot_state)
			trajectory.append(question)
			#A bot encodes the question, updates state, generates answer
			question_encoding = self.Abot.encode_question(question)
			a_history = self.Abot.encode_state_history(self.image_placeholder, self.caption_placeholder, question_encoding, abot_fact, abot_state)
			answer = self.Abot.get_answers(a_history)
			trajectory.append(answer)
			#Once answer is generated, fact is stored for use in next round of dialog (saved in self.fact)
			abot_fact = self.Abot.encode_fact(question, answer)
			#Encode the question answer pair and store in self.fact
			qbot_fact = self.Qbot.encode_fact(question, answer)
			#Add this fact to state_history
			qbot_state = self.Qbot.encode_state_history(qbot_state, qbot_fact)
			#Guess if needed
			if self.config.guess_every_round:
				guesses.append(self.Qbot.generate_image_representation) # TODO Make this proper method call
		#Final guess if not already added
		if not self.config.guess_every_round:
			guesses = self.Qbot.generate_image_representation
		return guesses

	def add_prediction_op(self, inputs, captions):
		guess = self.run_dialog(self.config.max_dialog_rounds)
		feed = self.create_feed_dict(inputs, captions)
		guesses = self.sess.run(guess, feed_dict = feed)
	
