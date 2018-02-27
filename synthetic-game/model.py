import numpy as np
from collections import defaultdict
from bots import SyntheticQBot
from bots import SyntheticABot
import os

class Dialog_Bots(object):
	def __init__(self, config):
		""" Sets up the configuration parameters, creates Q Bot + A Bot.
		"""
		self.config=config
		self.Qbot = SyntheticQBot(self.config.Q)
		self.Abot = SyntheticABot(self.config.A)
		self.eval_rewards = []

	def run_dialog(self, minibatch, batch_size, rounds_dialog=2):
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
		predictions = []
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
				predictions.append(self.Qbot.generate_image_predictions(q_bot_states))

		if not self.config.guess_every_round:
			predictions = self.Qbot.generate_image_predictions(q_bot_states)

		for i, q_bot_trajectory in enumerate(q_bot_trajectories):
			q_bot_trajectory.append((q_bot_states[i], -1))

		return predictions, q_bot_trajectories, a_bot_trajectories

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
			yield zip(images, captions, labels)

	def get_returns(self, trajectories, predictions, labels, gamma):
		""" Gets returns for a list of trajectories.
			+1 Reward if guess == answer
			-1 Otherwise
		"""
		all_returns = []
		all_rewards = []
		for i in xrange(len(predictions)):
			path_returns = [0] * len(trajectories[i])
			if predictions[i] == labels[i]:
				final_reward = 1.0
			else:
				final_reward = -1.0
			all_rewards.append(final_reward)
			discount_factors = np.power(gamma, np.arange(len(trajectories[i]), -1 ,-1))
			path_returns = final_reward * np.ones(len(trajectories[i])) * discount_factors
			all_returns.append(path_returns)
		return all_returns, all_rewards

	def train(self, batch_size=20, num_iterations=500, max_dialog_rounds=2):
		def apply_updates(bot, trajectories, all_returns, state_action_counts):
			""" Get Q-Learning updates that should be applied to a bot.
				We compute the running average of each state, action -> reward.
			"""
			for dialog_index, trajectory in enumerate(trajectories):  # for each dialog
				returns = all_returns[dialog_index]
				for timestep, (state, action) in enumerate(trajectory):
					state_action_counts[state][action] += 1
					state_action_count = state_action_counts[state][action]
					prev_q_val_contrib = (float(bot.Q[state][action]) / state_action_count) * (state_action_count - 1.0)
					cur_update_contrib = (1.0 / state_action_count) * returns[timestep]
					cur_q_val = prev_q_val_contrib + cur_update_contrib
					bot.Q[state][action] = cur_q_val

		def apply_regression_updates(bot, trajectories, predictions, all_returns):
			""" Get Q-Learning updates that should be applied to Q-bot's regression network.
				We set Q-regression-value of each state, action -> reward.
			"""
			for dialog_index, trajectory in enumerate(trajectories):  # for each dialog
				final_return = all_returns[dialog_index][-1]
				final_state, _ = trajectory[-1]
				final_action = predictions[dialog_index]
				bot.Q[final_state][final_action] = final_return

		average_rewards_across_training = []
		q_bot_state_action_counts = defaultdict(lambda: np.zeros(self.config.Q.num_actions))
		a_bot_state_action_counts = defaultdict(lambda: np.zeros(self.config.A.num_actions))
		minibatch_generator = self.get_minibatches(batch_size)
		for i in xrange(num_iterations):
			minibatch = minibatch_generator.next()
			images, captions, labels = [], [], []
			for (image, caption, label) in minibatch:
				images.append(image)
				captions.append(caption)
				labels.append(label)

			labels = [label for _, _, label in minibatch]
			predictions, q_bot_trajectories, a_bot_trajectories = self.run_dialog(minibatch, batch_size, max_dialog_rounds)
			update_q_bot = i % 2 == 0
			if update_q_bot:
				returns, rewards = self.get_returns(q_bot_trajectories, predictions, labels, self.config.Q.gamma)
				apply_updates(self.Qbot, q_bot_trajectories, returns, q_bot_state_action_counts)
				apply_regression_updates(self.Qbot, q_bot_trajectories, predictions, returns)
			else:
				returns, rewards = self.get_returns(a_bot_trajectories, predictions, labels, self.config.A.gamma)
				apply_updates(self.Abot, a_bot_trajectories, returns, a_bot_state_action_counts)
			avg_reward = np.mean(rewards)
			average_rewards_across_training.append(avg_reward)
			sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
			print "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
		
		def show_dialog(self, image,caption,answer,batch_size = 1, rounds_dialog=2):
			batch = (image,caption,-1)
			output, q_bot_trajectory, a_bot_trajectory = self.run_dialog(minibatch, batch_size=self.config.batch_size, rounds_dialog=2)
			print "FINAL PREDICTION = " + string(output)
			i=1
			while i<4:
				print "Qbot question:" + string(q_bot_trajectory[i])+"\n"
				print "Abot answer:" + string(a_bot_trajectory[i]) + "\n"
				i+=2
			
if __name__ == '__main__':
	db = Dialog_Bots(config)
	db.train()

