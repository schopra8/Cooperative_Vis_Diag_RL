import numpy as np
from collections import defaultdict
from bots import SyntheticQBot
from bots import SyntheticABot

class Dialog_Bots(object):
	def __init__(self, config):
		""" Sets up the configuration parameters, creates Q Bot + A Bot.
		"""
		self.config=config
		self.Qbot = SyntheticQBot(self.config.Q)
		self.Abot = SyntheticABot(self.config.A)

	def run_dialog(self, minibatch, batch_size=self.config.batch_size, rounds_dialog=2):
		""" Runs dialog for specified number of rounds:
				1) Q Bot asks question
				2) A Bot answers question based on history
				3) A Bot encodes question for later usage in it's history
				4) Q bot encodes the answer and updates state history
        Args:
			minibatch = [(image, caption, solution)]
			batch_size = num items in a batch
            rounds_dialog (int): Number of times this must repeat
        Returns:
            answer: The predictions of the Q bot at the end of (every) dialog
        """
		#First encode the caption and image in both bots
		images = [image for image, _, _ in minibatch]
		captions = [caption for _, caption, _ in minibatch]
		q_bot_states = self.Qbot.encode_captions(captions)
		a_bot_states  = self.Abot.encode_captions_images(captions, images)
		q_bot_trajectories = [[] for _ in xrange(batch_size)]
		a_bot_trajectories = [[] for _ in xrange(batch_size)]
		a_bot_recent_facts = [(-1, -1)] * batch_size # Sentinels for A Bot Fact 0
		q_bot_facts = []
		guesses = []
		for _ in xrange(rounds_dialog):
			questions = self.Qbot.get_questions(q_bot_states) # QBot generates questions (Q_t)
			for i, q in enumerate(questions): # Append to trajectory
				q_bot_trajectories[i].append((q_bot_states[i], q))

			question_encodings = self.Abot.encode_questions(questions) # ABot encodes questions (Q_t)
			a_bot_states = self.Abot.encode_state_histories(	# ABot encodes states (State, Y, Q_t, F_{t-1})
				images,
				question_encodings,
				a_bot_recent_facts,
				a_bot_states
			)
			answers = self.Abot.decode_answers(a_bot_states) # ABot generates answers (A_t)
			a_bot_recent_facts = self.Abot.encode_facts(question_encodings, answers) # ABot generates facts (F_t)
			# TODO: How do we account for the first state (image, caption)? It doesn't yield an action.
			for i, a in enumerate(answers): # Append to trajectory
				a_bot_trajectories[i].append((a_bot_states[i], a))

			q_bot_facts = self.Qbot.encode_facts(questions, answers) # QBot encodes facts (F_t)
			q_bot_states = self.Qbot.encode_state_histories(q_bot_states, q_bot_facts) # QBot encode states
			if self.config.guess_every_round:
				guesses.append(self.Qbot.generate_image_representations(q_bot_states))

		if not self.config.guess_every_round:
			guesses = self.Qbot.generate_image_representations(q_bot_states)

		return guesses, q_bot_trajectories, a_bot_trajectories

	def get_minibatches(self, batch_size=20):
		# TODO Implement batching of captions, images
		pass

	def get_returns(self, trajectories, guesses, answers, gamma):
		""" Gets returns for a list of trajectories.
			+1 Reward if guess == answer
			-1 Otherwise
		"""
		# TODO: Confirm the reward structure
		all_returns = []
		for i, g in enumerate(guesses):
			path_returns = [0] * len(trajectories[i])
			if g == answers[i]:
				final_reward = 1.0
			else:
				final_reward = -1.0
			discount_factors = np.power(gamma, np.arange(len(trajectories[i])))
			path_returns = list(np.multiply(final_reward * np.ones(len(trajectories[i]), discount_factors)))
			all_returns.append(path_returns)
		return all_returns

	def train(self, batch_size=20, num_iterations=500, max_dialog_rounds=1):
		def apply_updates(bot, trajectories, returns, state_action_counts):
			""" Get Q-Learning updates that should be applied to a bot.
				We compute the running average of each state, action -> reward.
			"""
			for t_index, t in enumerate(trajectories):
				t_returns = returns[t_index]
				for r_index, (state, action) in enumerate(t):
					state_action_counts[state][action] += 1
					state_action_count = state_action_counts[state][action]
					prev_q_val_contrib = ((bot.Q[state][action] + 0.0) / state_action_count) * (state_action_count - 1.0)
					cur_update_contrib = (1.0 / state_action_count) * t_returns[r_index]
					cur_q_val = prev_q_val_contrib + cur_update_contrib
					bot.Q[state][action] = cur_q_val

		q_bot_state_action_counts = defaultdict(lambda: np.zeros())
		a_bot_state_action_counts = defaultdict(lambda: np.zeros())
		for i in num_iterations:
			minibatch = self.get_minibatches(batch_size)
			answers = [answer for _, _, answer in minibatch]
			guesses, q_bot_trajectories, a_bot_trajectories = self.run_dialog(minibatch, batch_size, max_dialog_rounds)
			update_q_bot = i % 2 == 0
			if update_q_bot:
				returns = self.get_returns(q_bot_trajectories, guesses, answers, self.config.Q.gamma)
				apply_updates(self.Qbot, q_bot_trajectories, returns, q_bot_state_action_counts)
			else:
				returns = self.get_returns(a_bot_trajectories, guesses, answers, self.config.A.gamma)
				apply_updates(self.Abot, a_bot_trajectories, returns, a_bot_state_action_counts)





