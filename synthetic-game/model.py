import numpy as np
from collections import defaultdict
from bots import SyntheticQBot
from bots import SyntheticABot
import os
import matplotlib.pyplot as plt
import scipy.signal as sig

class Dialog_Bots(object):
	def __init__(self, config):
		""" Sets up the configuration parameters, creates Q Bot + A Bot.
		"""
		self.config = config
		self.Qbot = SyntheticQBot(self.config.Q)
		self.Abot = SyntheticABot(self.config.A)
		self.eval_rewards = []

	def run_dialog(self, images, captions, num_dialog_rounds=2, test = False):
		""" Runs dialog for specified number of rounds:
				1) Q Bot asks question
				2) A Bot answers question based on history
				3) A Bot encodes question for later usage in it's history
				4) Q bot encodes the answer and updates state history
        Args:
			images = [array(image)]
			captions = [caption]
			batch_size = num items in a batch
            num_dialog_rounds (int): Number of times this must repeat
        Returns:
            answer: The predictions of the Q bot at the end of (every) dialog
        """
		#First encode the caption and image in both bots
		batch_size = len(images)
		images = map(tuple, images)
		q_bot_states = self.Qbot.encode_captions(captions)
		a_bot_states  = self.Abot.encode_images_captions(images, captions)
		q_bot_trajectories = [[] for _ in xrange(batch_size)]
		a_bot_trajectories = [[] for _ in xrange(batch_size)]
		a_bot_recent_facts = [(-1, -1)] * batch_size # Sentinels for A Bot Fact 0
		q_bot_facts = []
		predictions = []
		for _ in xrange(num_dialog_rounds):
			if test is False:
				questions = self.Qbot.get_questions(q_bot_states, self.config.Q.epsilon) # QBot generates questions (Q_t)
			else:
				questions = self.Qbot.get_questions(q_bot_states)
			for i, q in enumerate(questions): # Append to trajectory
				q_bot_trajectories[i].append((q_bot_states[i], q))

			question_encodings = self.Abot.encode_questions(questions) # ABot encodes questions (Q_t)
			a_bot_states = self.Abot.encode_state_histories(	# ABot encodes states (State, Y, Q_t, F_{t-1})
				images,
				question_encodings,
				a_bot_recent_facts,
				a_bot_states
			)
			if test is False:
				answers = self.Abot.get_answers(a_bot_states, self.config.A.epsilon) # ABot generates answers (A_t)
			else:
				answers = self.Abot.get_answers(a_bot_states)
			a_bot_recent_facts = self.Abot.encode_facts(question_encodings, answers) # ABot generates facts (F_t)
			for i, a in enumerate(answers): # Append to trajectory
				a_bot_trajectories[i].append((a_bot_states[i], a))

			q_bot_facts = self.Qbot.encode_facts(questions, answers) # QBot encodes facts (F_t)
			q_bot_states = self.Qbot.encode_state_histories(q_bot_states, q_bot_facts) # QBot encode states
			if self.config.guess_every_round:
				if test is False:
					predictions.append(self.Qbot.get_image_predictions(q_bot_states, self.config.Q.epsilon))
				else:
					predictions.append(self.Qbot.get_image_predictions(q_bot_states))
		if not self.config.guess_every_round:
			if test is False:
				predictions = self.Qbot.get_image_predictions(q_bot_states, self.config.Q.epsilon)
			else:
				predictions = self.Qbot.get_image_predictions(q_bot_states)

		for i, q_bot_trajectory in enumerate(q_bot_trajectories):
			q_bot_trajectory.append((q_bot_states[i], -1))

		return predictions, q_bot_trajectories, a_bot_trajectories

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

	def get_minibatches(self, batch_size=20):
		data = np.loadtxt(os.path.join(self.config.DATA_DIR, self.config.DATA_FILE), skiprows=1, delimiter=',')
		np.random.shuffle(data)
		caption_lookup = {0: [0,1], 1: [0,2], 2:[1,0], 3:[1,2], 4: [2,0], 5:[2,1]}
		i = 0
		size = data.shape[0]
		while True:
			batch_data = data[i%size:(i%size+batch_size),:]
			images = batch_data[:,:3]
			captions = batch_data [:,3]
			labels = batch_data[:,4]
			i += batch_size
			yield images, captions, labels
			

	def get_returns(self, trajectories, predictions, labels, gamma):
		""" Gets returns for a list of trajectories.
			+1 Reward if guess == answer
			-1 Otherwise
		"""
		all_returns = []
		all_rewards = []
		for i in xrange(len(predictions)):
			path_returns = [0] * len(trajectories[i])
			# print "predictions[i], labels[i]", predictions[i], labels[i]
			if predictions[i] == labels[i]:
				# print 'CORRECT PREDICTION MADE!'
				final_reward = 1.0
			else:
				final_reward = -1.0
			all_rewards.append(final_reward)
			discount_factors = np.power(gamma, np.arange(len(trajectories[i])-1, -1, -1))
			path_returns = final_reward * np.ones(len(trajectories[i])) * discount_factors
			all_returns.append(path_returns)
		return all_returns, all_rewards

	def train(self, batch_size=20, num_iterations=500, max_dialog_rounds=2):
		def calculate_running_average(updated_count, prev_q_value, new_return):
			prev_q_val_contrib = (float(prev_q_value) / updated_count) * (updated_count - 1.0)
			cur_update_contrib = (1.0 / updated_count) * new_return
			new_q_val = prev_q_val_contrib + cur_update_contrib
			return new_q_val

		def apply_updates(bot, trajectories, all_returns, state_action_counts):
			""" Get Q-Learning updates that should be applied to a bot.
				We compute the running average of each state, action -> reward.
			"""
			for dialog_index, trajectory in enumerate(trajectories):  # for each dialog
				returns = all_returns[dialog_index]
				for timestep, (state, action) in enumerate(trajectory):
					if action == -1: continue
					state_action_counts[state][action] += 1
					state_action_count = state_action_counts[state][action]
					# print "state, action", state, action
					# print "Q value", bot.Q[state][action]
					new_q_val = calculate_running_average(state_action_count, bot.Q[state][action], returns[timestep])
					bot.Q[state][action] = new_q_val
					# print "Q value", bot.Q[state][action]

		def apply_regression_updates(bot, trajectories, predictions, all_returns, state_action_counts):
			""" Get Q-Learning updates that should be applied to Q-bot's regression network.
				We set Q-regression-value of each state, action -> reward.
			"""
			for dialog_index, trajectory in enumerate(trajectories):  # for each dialog
				final_return = all_returns[dialog_index][-1]
				final_state, _ = trajectory[-1]
				final_action = predictions[dialog_index]

				state_action_counts[final_state][final_action] += 1
				state_action_count = state_action_counts[final_state][final_action]

				new_q_val = calculate_running_average(state_action_count, bot.Q_regression[final_state][final_action], final_return)
				bot.Q_regression[final_state][final_action] = new_q_val
				# print "final_state, final_action", final_state, final_action
				# print "final Q value", bot.Q_regression[final_state][final_action]
				# bot.Q_regression[final_state][final_action] = final_return
				# if final_return == 1.0:
				# 	print final_state
				# 	print bot.Q_regression[final_state]
				# print "final Q value", bot.Q_regression[final_state][final_action]

		self.average_rewards_across_training = []
		q_bot_state_action_counts = defaultdict(lambda: np.zeros(self.config.Q.num_actions))
		q_bot_final_state_action_counts = defaultdict(lambda: np.zeros(self.config.Q.num_classes))
		a_bot_state_action_counts = defaultdict(lambda: np.zeros(self.config.A.num_actions))
		minibatch_generator = self.get_minibatches(batch_size)
		test_minibatch_generator = self.get_minibatches(self.config.test_batch)
		self.eval_rewards = []
		for i in xrange(num_iterations):
			images, captions, labels = minibatch_generator.next()
			predictions, q_bot_trajectories, a_bot_trajectories = self.run_dialog(images, captions, max_dialog_rounds)
			self.update_epsilon(i+1)
			update_q_bot = i % 2 == 0
			if update_q_bot:
				returns, rewards = self.get_returns(q_bot_trajectories, predictions, labels, self.config.Q.gamma)
				# print self.Qbot.Q
				apply_updates(self.Qbot, q_bot_trajectories, returns, q_bot_state_action_counts)
				apply_regression_updates(self.Qbot, q_bot_trajectories, predictions, returns, q_bot_final_state_action_counts)
				# print self.Qbot.Q
				# print self.Qbot.Q_regression
			else:
				returns, rewards = self.get_returns(a_bot_trajectories, predictions, labels, self.config.A.gamma)
				apply_updates(self.Abot, a_bot_trajectories, returns, a_bot_state_action_counts)
			# TODO: evaluate on: simulate new dialogs with epsilon = 1, so no exploration
			avg_reward = np.mean(rewards)
			self.average_rewards_across_training.append(avg_reward)
			sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
			if self.config.verbose:
				print "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
			
			#Evaluate with evaluation epsilon over a batch and store average rewards
			if i%self.config.eval_every == 0:
				if self.config.verbose:
					print "Evaluating..."
				images, captions, labels = test_minibatch_generator.next()
				predictions, q_bot_trajectory, a_bot_trajectory = self.run_dialog(images, captions, max_dialog_rounds, test= True)
				_, rewards = self.get_returns(q_bot_trajectory, predictions, labels, self.config.Q.gamma)
				avg_eval_reward = np.mean(rewards)
				self.eval_rewards.append(avg_eval_reward)
				sigma_eval_reward = np.sqrt(np.var(rewards) / len(rewards))
				if self.config.verbose:
					print "Evaluation reward: {:04.2f} +/- {:04.2f}".format(avg_eval_reward, sigma_eval_reward)



		# print self.Qbot.Q_regression

	def show_dialog(self, image, caption, answer):
		output, q_bot_trajectory, a_bot_trajectory = self.run_dialog(image,caption, test=True)
		for i in xrange(image.shape[0]):
			print "IMAGE is:" + str(image[i])
			print "CAPTION is:" +str(self.config.caption_lookup[caption[i]])
			print "GROUND TRUTH = " + str([int(answer[i]/12),answer[i]%12])
			print "FINAL PREDICTION = " + str([int(output[i]/12), output[i]%12])
			j=0
			while j<2:
				print "Qbot question:" + str(self.config.Q.q_bot_lookup[q_bot_trajectory[i][j][1]])+"\n"
				print "Abot answer:" + str(a_bot_trajectory[i][j][1]) + "\n"
				j+=1
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
