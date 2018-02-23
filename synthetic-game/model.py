import tensorflow as tf
import numpy as np
from bots import SyntheticQBot
from bots import SyntheticABot

class Dialog_Bots(object):
	def __init__(self,config):
		#Sets up the configuration parameters
		self.config=config
		#Adds a Qbot
		self.Qbot = SynethicQBot(self.config.Q)
		#Adds an A bot
		self.Abot = SyntheticABot(self.config.A)
	def add_placeholders(self):
		"""Adds placeholders to the graph
        """
		#Inputs are the images/image representations
		self.image_placeholder = tf.placeholder(dtype = tf.int32, shape = [None, self.config.input_dimension])
		#Captions are vectors (Design choice: Vocabulary of A bot or different?)
		self.caption_placeholder = tf.placeholder(dtype = tf.int32, shape = [None, self.config.caption_dimension])
	def create_feed_dict(self, inputs, captions):
		#Feed in image and captions
		feed_dict = { self.image_placeholder : inputs, self.caption_placeholder : captions
		}
		return feed_dict
	def run_dialog(self, rounds_dialog = 2):
		"""Runs dialog for specified number of rounds: Starts with Q-bot asking a question, A-bot answering the question and encoding the information
			Q bot encoding the answer and updating state history
        Args:
            rounds_dialog (int): Number of times this must repeat
        Returns:
            answer: The predictions of the Q bot at the end of (every) dialog
        """
		#First encode the caption and image in both bots
		self.Qbot.encode_caption(self.caption_placeholder)
		self.Abot.encode_caption_image(self.caption_placeholder, self.image_placeholder)
		guesses=[]
		for i in range(rounds_dialog-1):
			#First question asked by Q bot
			question = self.Qbot.decode_question()
			#A bot encodes the question, updates state, generates answer
			self.Abot.encode_question(question)
			self.Abot.encode_state_history(self.Abot.question_encoding)
			answer = self.Abot.decode_answer(self.Abot.state)
			#Once answer is generated, fact is stored for use in next round of dialog (saved in self.fact)
			self.Abot.encode_fact(question, answer)
			#Encode the question answer pair and store in self.fact
			self.Qbot.encode_fact(question, answer)
			#Add this fact to state_history
			self.Qbot.encode_state_history()
			#Guess if needed
			if self.config.guess_every_round:
				guesses.append(self.Qbot.generate_image_representation)
		#Final guess if not already added
		if self.config.guess_every_round is False:
			guesses = self.Qbot.generate_image_representation
		return guesses

	def add_prediction_op(self, inputs, captions):
		guess = self.run_dialog(self.config.max_dialog_rounds)
		feed = self.create_feed_dict(inputs, captions)
		guesses = sess.run(guess, feed_dict = feed)
	
