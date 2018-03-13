import numpy as np
from collections import defaultdict
from bots import SyntheticQBot
from bots import SyntheticABot
import math, os
import matplotlib.pyplot as plt
import scipy.signal as sig

import pdb

class Dialog_Bots(object):
	def __init__(self, config):
		""" Sets up the configuration parameters. Create ABot and QBot.
		"""
		self.config = config
		self.Qbot = SyntheticQBot(self.config.Q)
		self.Abot = SyntheticABot(self.config.A)
		self.eval_rewards = []

	def run_dialog(self, images, game_types, num_dialog_rounds=2, test=False):
		""" Runs dialog for specified number of rounds:
				1) Q Bot asks question
				2) A Bot answers question based on history
				3) A Bot encodes question for later usage in it's history
				4) Q bot encodes the answer and updates state history
        Args:
			images = [array(image)]
			game_types = [game_type]
			batch_size = num items in a batch
            num_dialog_rounds (int): Number of times this must repeat
        Returns:
            answer: The predictions of the Q bot at the end of (every) dialog
        """
		batch_size = len(images)
		images = map(tuple, images)
		prev_answers = []

		# Determine epsilon
		if test:
			q_epsilon = 1.0
			a_epsilon = 1.0
		else:
			q_epsilon = self.config.q_epsilon
			a_epsilon = self.config.a_epsilon

		# Simulate Rounds of Dialog
		self.Qbot.set_initial_states(game_types)
		for round in xrange(num_dialog_rounds):
			questions = self.Qbot.decode_questions(q_epsilon)
			if round == 0:
				self.Abot.set_initial_states(game_types, images, questions)
			else:
				self.Abot.update_states(prev_answers, questions)
			answers = self.Abot.decode_answers(a_epsilon)
			self.Qbot.update_states(questions, answers)
			prev_answers = answers
		
		# Store last answers in A bot state
		null_questions = [-1] * batch_size
		self.Abot.update_states(prev_answers, null_questions) 
		
		# Generate Predictions
		predictions = self.Qbot.generate_image_predictions(q_epsilon)

		# Generate Trajectories 
		q_bot_trajectories = []
		a_bot_trajectories = []
		for state in self.Qbot.states:
			trajectory = [(state[0:i], state[i][0]) for i in xrange(1, len(state))]
			q_bot_trajectories.append(trajectory)

		for state in self.Abot.states:
			trajectory = [(state[0:i], state[i][0]) for i in xrange(1, len(state))]
			a_bot_trajectories.append(trajectory)

		return predictions, q_bot_trajectories, a_bot_trajectories

	def get_minibatches(self, batch_size=10000):
		data = np.loadtxt(os.path.join(self.config.DATA_DIR, self.config.DATA_FILE), skiprows=1, delimiter=',')
		np.random.shuffle(data)
		i = 0
		size = data.shape[0]
		num_copies = int(batch_size / size) + 1
		while True:
			if num_copies > 1:
				images = np.repeat(data[:,:3], num_copies, axis=0)
				game_states = np.repeat(data[:,3], num_copies)
				labels = np.repeat(data[:,4], num_copies)
			else:
				batch_data = data[i%size:(i%size+batch_size),:]
				images = batch_data[:,:3]
				game_states = batch_data [:,3]
				labels = batch_data[:,4]
				i += batch_size
			yield images, game_states, labels

	def get_final_rewards(self, predictions, labels):
		""" Gets final rewards for a list of trajectories.
			+1 Reward if guess == answer
			-1 Otherwise
		"""
		return [1 if predictions[i] == labels[i] else -1 for i in xrange(len(predictions))]

	def get_returns(self, trajectories, predictions, labels):
		""" Gets returns for a list of trajectories.
			+1 Reward if guess == answer
			-1 Otherwise
		"""
		all_returns = []
		for i in xrange(len(predictions)):
			path_returns = [0] * len(trajectories[i])
			if predictions[i] == labels[i]:
				final_reward = 1.0
			else:
				final_reward = -1.0
			path_returns = final_reward * np.ones(len(trajectories[i]))
			all_returns.append(path_returns)
		return all_returns

	def evaluate(self, minibatch_generator, max_dialog_rounds):
		if self.config.verbose:
			print "Evaluating..."
		images, game_states, labels = minibatch_generator.next()
		predictions, _, _ = self.run_dialog(images, game_states, max_dialog_rounds, test=True)
		rewards = self.get_final_rewards(predictions, labels)
		avg_eval_reward = np.mean(rewards)
		self.eval_rewards.append(avg_eval_reward)
		sigma_eval_reward = np.sqrt(np.var(rewards) / len(rewards))
		if self.config.verbose:
			print "Evaluation reward: {:04.2f} +/- {:04.2f}".format(avg_eval_reward, sigma_eval_reward)

	def show_dialog(self, images, game_states, answers):
		output, q_bot_trajectory, a_bot_trajectory = self.run_dialog(images, game_states, test=True)
		for i in xrange(images.shape[0]):
			print "Image is:" + str(images[i])
			print "Caption is:" +str(self.config.game_states_lookup[game_states[i]])
			print "GROUND TRUTH = " + str([int(answers[i]/12),answers[i]%12])
			print "FINAL PREDICTION = " + str([int(output[i]/12), output[i]%12])
			for j in xrange(2):
				print "Qbot question:" + str(self.config.Q.q_bot_lookup[q_bot_trajectory[i][j][1]])+"\n"
				print "Abot answer:" + str(a_bot_trajectory[i][j][1]) + "\n"
				
	def generate_graphs(self):
		#Smoothing of data
		self.average_rewards_across_training = sig.savgol_filter(self.average_rewards_across_training, self.config.win_length, self.config.polyorder)
		self.eval_rewards = sig.savgol_filter(self.eval_rewards, self.config.win_length, self.config.polyorder)
		#Plotting
		plt.figure(1)
		plt.plot(self.average_rewards_across_training, 'k',label = "Training rewards")
		plt.xlabel("Iteration Number")
		plt.ylabel("Rewards")
		plt.savefig(os.path.join(self.config.output_dir,'training_rewards.png'), bbox_inches = 'tight')
		plt.figure(2)
		plt.plot(self.eval_rewards, 'r', label = "Evaluations Rewards")
		plt.xlabel("Evaluation episode Number")
		plt.ylabel("Rewards")
		plt.savefig(os.path.join(self.config.output_dir,'evaluation_rewards.png'), bbox_inches = 'tight')


	def train(self, batch_size=10000, num_iterations=500, max_dialog_rounds=2):
		def calculate_running_average(updated_count, prev_q_value, new_return):
			prev_q_val_contrib = (float(prev_q_value) / updated_count) * (updated_count - 1.0)
			cur_update_contrib = float(new_return)/updated_count
			new_q_val = prev_q_val_contrib + cur_update_contrib
			return new_q_val

		def apply_updates(bot, trajectories, all_returns, state_action_counts):
			""" Get Q-Learning updates that should be applied to a bot.
				We compute the running average of each state, action -> reward.
			"""
			for dialog_index, trajectory in enumerate(trajectories):  # for each dialog
				returns = all_returns[dialog_index]
				for timestep, (state, action) in enumerate(trajectory):
					state_action_counts[state][action] += 1
					state_action_count = state_action_counts[state][action]
					new_q_val = calculate_running_average(state_action_count, bot.Q[state][action], returns[timestep])
					bot.Q[state][action] = new_q_val

		def apply_regression_updates(bot, predictions, all_returns, state_action_counts):
			""" Get Q-Learning updates that should be applied to Q-bot's regression network.
				We set Q-regression-value of each state, action -> reward.
			"""
			for dialog_index in xrange(len(predictions)):  # for each dialog
				final_return = all_returns[dialog_index][-1]
				final_state = bot.states[dialog_index]
				final_action = predictions[dialog_index]

				# print "FINAL STATE: {}".format(final_state)
				# print "FINAL_ACTION: {}".format(final_action)
				# print bot.Q_regression[final_state]

				state_action_counts[final_state][final_action] += 1
				state_action_count = state_action_counts[final_state][final_action]

				new_q_val = calculate_running_average(state_action_count, bot.Q_regression[final_state][final_action], final_return)
				bot.Q_regression[final_state][final_action] = new_q_val

				# print bot.Q_regression[final_state]

		self.average_rewards_across_training = []
		self.eval_rewards = []

		q_bot_state_action_counts = defaultdict(lambda: np.zeros(self.config.Q.num_actions))
		q_bot_final_state_action_counts = defaultdict(lambda: np.zeros(self.config.Q.num_classes))
		a_bot_state_action_counts = defaultdict(lambda: np.zeros(self.config.A.num_actions))
		minibatch_generator = self.get_minibatches(batch_size)
		test_minibatch_generator = self.get_minibatches(self.config.test_batch)

		for i in xrange(num_iterations):
			images, game_states, labels = minibatch_generator.next()

			# print "IMAGES: {}".format(images)
			# print "GAME STATES: {}".format(game_states)
			# print "LABELS: {}".format(labels)

			predictions, q_bot_trajectories, a_bot_trajectories = self.run_dialog(images, game_states, max_dialog_rounds)
			rewards = self.get_final_rewards(predictions, labels)

			# print "PREDICTIONS: {}".format(predictions)
			# print "Q Bot Trajectories: {}".format(q_bot_trajectories)
			# print "A Bot Trajectories: {}".format(a_bot_trajectories)
			# print "Q Bot States {}".format(self.Qbot.states)
			# print "A Bot States {}".format(self.Abot.states)
			# print "Rewards: {}".format(rewards)
			
			# self.update_epsilon(i+1)
			update_q_bot = i % 2 == 0
			if update_q_bot:
				returns = self.get_returns(q_bot_trajectories, predictions, labels)
				apply_updates(self.Qbot, q_bot_trajectories, returns, q_bot_state_action_counts)
				apply_regression_updates(self.Qbot, predictions, returns, q_bot_final_state_action_counts)
			else:
				returns = self.get_returns(a_bot_trajectories, predictions, labels)
				apply_updates(self.Abot, a_bot_trajectories, returns, a_bot_state_action_counts)

			avg_reward = np.mean(rewards)
			self.average_rewards_across_training.append(avg_reward)
			sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
			if self.config.verbose:
				print "Average reward at iteration {}: {:04.2f} +/- {:04.2f}".format(i, avg_reward, sigma_reward)
			
			#Evaluate with evaluation epsilon over a batch and store average rewards
			if i%self.config.eval_every == 0:
				self.evaluate(test_minibatch_generator, max_dialog_rounds)

	# ----------------
	def update_epsilon(self, iteration_num):
		if iteration_num > self.config.Q.iterations:
			self.config.Q.epsilon = self.config.Q.epsilon_end
			
		else:
			self.config.Q.epsilon = self.config.Q.epsilon_start - (iteration_num *(self.config.Q.epsilon_start-
															self.config.Q.epsilon_end)/self.config.Q.iterations)
		
		if iteration_num >self.config.A.iterations:
			self.config.A.epsilon = self.config.A.epsilon_end
		else:
			self.config.A.epsilon = self.config.A.epsilon_start - (iteration_num *(self.config.A.epsilon_start-
															self.config.A.epsilon_end)/self.config.A.iterations)