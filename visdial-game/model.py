import tensorflow as tf
import numpy as np
from bots import DeepQBot
from bots import DeepABot

class model():
	def __init__(self, config):
		""" Sets up the configuration parameters, creates Q Bot + A Bot.
		"""
		self.config= config
		self.embedding_matrix = tf.get_variable("word_embeddings", shape=[self.config.VOCAB_SIZE, self.config.EMBEDDING_SIZE])
		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		self.Qbot = DeepQBot(
			self.config,
			tf.nn.embedding_lookup(self.embedding_matrix, ids=tf.Variable([self.config.START_TOKEN_IDX])),
			self.embedding_matrix
		)
		self.Abot = DeepABot(
			self.config,
			tf.nn.embedding_lookup(self.embedding_matrix, ids=tf.Variable([self.config.START_TOKEN_IDX])),
			self.embedding_matrix
		)
		self.add_placeholders()
		self.add_all_ops()
		self.add_update_op()
        self.best_model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()
		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.config.keep)

	def add_placeholders(self):
		"""
		Adds placeholders needed for training
		"""
		self.images = tf.placeholder(tf.float32, shape = [None, self.config.IMG_REP_DIM])
		self.captions = tf.placeholder(tf.int32, shape = [None, self.config.MAX_CAPTION_LENGTH])
		self.caption_lengths = tf.placeholder(tf.int32, shape = [None])
		self.true_questions = tf.placeholder(tf.int32, shape = [None, self.config.number_of_dialog_rounds, self.config.MAX_QUESTION_LENGTH])
		self.true_answers = tf.placeholder(tf.int32, shape = [None, self.config.number_of_dialog_rounds, self.config.MAX_ANSWER_LENGTH])
		self.true_question_lengths = tf.placeholder(tf.int32, shape = [None, self.config.number_of_dialog_rounds])
		self.true_answer_lengths = tf.placeholder(tf.int32, shape = [None, self.config.number_of_dialog_rounds])
		self.supervised_learning_rounds = tf.placeholder(tf.int32, shape = [])

	def add_update_op(self):
		"""
		Add update_op to perform one gradient descent step with clipped gradients
		"""
		optimizer = tf.train.AdamOptimizer(learning_rate = self.config.learning_rate)
		grads, variables = optimizer.compute_gradients(self.loss)
		clipped_grads = tf.clip_by_global_norm(grads, self.config.max_gradient_norm)
		self.update_op = optimizer.apply_gradients(zip(clipped_grads, variables), global_step = self.global_step)
	
	def add_all_ops(self):
		self.loss, self.generated_questions, self.generated_answers, self.generated_images, self.batch_rewards = self.run_dialog()
		tf.summary.scalar('loss_start', self.loss)
		tf.summary.scalar('batch_rewards', np.mean(np.asarray(self.batch_rewards)))

	
	def run_dialog(self):
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
		self.captions = tf.nn.embedding_lookup(self.embedding_matrix, self.captions)
		Q_state = self.Qbot.encode_captions(self.captions, self.caption_lengths)
		A_state = self.Abot.encode_images_captions(self.captions, self.images, self.caption_lengths)
		A_fact = self.Abot.encode_facts(self.captions, self.caption_lengths)
		prev_image_guess = self.Qbot.generate_image_representations(Q_state)
		image_loss = 0
		loss = 0
		generated_questions = []
		generated_answers = []
		generated_images = []
		batch_rewards = []
		for i in xrange(self.config.num_dialog_rounds):
			x = tf.constant(i)
			if tf.greater_equal(x, self.supervised_learning_rounds): ## RL training
				#Q-Bot generates question logits
				question_logits, question_lengths = self.Qbot.get_questions(Q_state, supervised_training = False)
				#Find embeddings of questions
				questions = tf.nn.embedding_lookup(self.embedding_matrix, tf.argmax(question_logits, axis = 2))
				generated_questions.append(questions)
				#A-bot encodes questions
				encoded_questions = self.Abot.encode_questions(questions)
				#A-bot updates state
				A_state = self.Abot.encode_state_histories(A_fact, self.images, encoded_questions, A_state)
				#Abot generates answer logits
				answer_logits, answer_lengths = self.Abot.get_answers(A_state, supervised_training=False)
				#Generate facts for that round of dialog
				generated_answers.append(tf.nn.embedding_lookup(self.embedding_matrix, tf.argmax(answer_logits, axis = 2)))

				facts, fact_lengths = self.concatenate_q_a(question_logits, question_lengths, answer_logits, answer_lengths)
				#Embed the facts into word vector space
				facts = tf.nn.embedding_lookup(self.embedding_matrix, tf.argmax(facts, axis = 2))
				#Encode facts into both bots
				A_fact = self.Abot.encode_facts(facts, fact_lengths)
				Q_fact = self.Qbot.encode_facts(facts, fact_lengths)
				Q_state = self.Qbot.encode_state_histories(Q_fact, Q_state)
				# QBot Generates guess of image
				image_guess = self.Qbot.generate_image_representations(Q_state)
				generated_images.append(image_guess)
				#Calculate loss for this round
				rewards = tf.reduce_sum(tf.square(prev_image_guess - self.images), axis = 1) - tf.reduce_sum(tf.square(image_guess - self.images), axis = 1)
				batch_rewards.append(tf.reduce_mean(rewards))
				prev_image_guess = image_guess

				#### CHANGE HERE FOR UPDATING ONLY SINGLE BOT
				prob_questions = tf.argmax(tf.nn.softmax(question_logits), axis = 2)
				prob_answers = tf.argmax(tf.nn.softmax(answer_logits), axis = 2)
				negative_log_prob_exchange = -tf.log(prob_questions)-tf.log(prob_answers)
				loss += tf.reduce_mean(negative_log_prob_exchange*rewards)

			else: ## Supervised training
				## ACCESS TRUE QUESTIONS  AND ANSWERS FOR THIS ROUND OF DIALOG
				questions = self.true_questions[:,i,:]
				answers = self.true_answers[:,i,:]
				true_question_lengths_round = self.true_question_lengths[:,i]
				true_answer_lengths_round = self.true_answer_lengths[:,i]
				#Generate questions based on current state
				question_logits, question_lengths = self.Qbot.get_questions(
					Q_state,
					true_questions=questions,
					true_question_lengths=true_question_lengths_round,
					supervised_training=True
				)
				#Encode the true questions
				encoded_questions = self.Abot.encode_questions(tf.nn.embedding_lookup(self.embedding_matrix, questions))
				#Update A state based on true question
				A_state = self.Abot.encode_state_histories(A_fact, self.images, encoded_questions, A_state)
				# ABot Generates answers based on current state
				answer_logits, answer_lengths = self.Abot.get_answers(
					A_state,
					true_answers=answers,
					true_answer_lengths=answer_lengths,
					supervised_training=True
				)
				#Generate facts from true questions and answers
				facts, fact_lengths = self.concatenate_q_a(questions, true_question_lengths_round, answers, true_answer_lengths_round)
				facts = tf.nn.embedding_lookup(self.embedding_matrix, facts)
				A_fact = self.Abot.encode_facts(facts, fact_lengths)
				Q_fact = self.Qbot.encode_facts(facts, fact_lengths)
				#Update state histories using current facts
				Q_state = self.Qbot.encode_state_histories(Q_state, facts, fact_lengths)
				#Guess image
				image_guess = self.Qbot.generate_image_representations(Q_state)
				#### Loss for supervised training
				dialog_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = question_logits, labels = questions)
				dialog_loss += tf.nn.sparse_softmax_cross_entropy_with_logits(logits = answer_logits, labels = answers)
				image_loss += tf.nn.l2_loss(image_guess - self.images)
				loss += dialog_loss + image_loss
		return loss, generated_questions, generated_answers, generated_images, batch_rewards

	def get_minibatches(self, batch_size=40):
		pass

	def train(self, sess, num_epochs = 400, batch_size=20):
		summary_writer = tf.summary.FileWriter(self.config.FLAGS.train_dir, sess.graph)

		curriculum = 0
		for i in xrange(num_epochs):
			if i<15:
				curriculum = 10
			else:
				curriculum -= 1
			if curriculum <0:
				curriculum = 0
			for batch in generate_minibatches(self.config.batch_size):
				loss = self.train_on_batch(sess, batch, summary_writer, supervised_learning_rounds = curriculum)
				if self.global_step % self.config.eval_every == 0:
					dev_loss, dev_MRR = self.evaluate(sess, i, i == 14)
					self.write_summary(dev_loss, "dev/loss_total", summary_writer, self.global_step)
					self.write_summary(dev_MRR, "dev/MRR_total", summary_writer, self.global_step)
	
	def train_on_batch(self, sess, batch, summary_writer, supervised_learning_rounds = 10):
		images, captions, caption_lengths, true_questions, true_question_lengths, true_answers, true_answer_lengths = batch

		feed = {
			self.images:images,
			self.captions:captions,
			self.caption_lengths:caption_lengths,
			self.true_questions:true_questions,
			self.true_question_lengths:true_question_lengths,
			self.true_answers:true_answers,
			self.true_answer_lengths: true_answer_lengths,
			self.supervised_learning_rounds:supervised_learning_rounds
		}
		summary, _, global_step, loss, rewards = sess.run([self.summaries, self.update_op, self.global_step, self.loss, self.batch_rewards], feed_dict = feed)
		summary_writer.add_summary(summary, global_step)
		return loss, rewards
	
	def write_summary(self, value, tag, summary_writer, global_step):
		""" Write a single summary value to tensorboard
		"""
    	summary = tf.Summary()
    	summary.value.add(tag=tag, simple_value=value)
    	summary_writer.add_summary(summary, global_step)
	
	def evaluate(self, sess, epoch, compute_MRR = False):
		dev_loss = 0
		for batch in generate_dev_minibatches(self.config.batch_size):
			true_images, _, _, _, _, _, gt_indices = batch
			loss, preds, gen_answers, gen_questions = self.eval_on_batch(sess, batch)
			dev_loss += loss
			MRR = np.zeros([self.config.number_of_dialog_rounds])
			if compute_MRR:
				for round_number, p in enumerate(preds):
					percentage_rank_gt = self.compute_mrr(p, gt_indices, true_images, round_number, epoch)
					MRR[round_number] += tf.reduce_mean(percentage_rank_gt)
		return dev_loss, MRR

	def eval_on_batch(self, sess, batch):
		images, captions, caption_lengths, _, _, _, _ = batch
		feed = {
			self.images:images,
			self.captions:captions,
			self.caption_lengths: caption_lengths
			self.supervised_learning_rounds:0
		}
		loss, images, answers, questions, rewards = sess.run([self.loss, self.generated_images, self.generated_answers, self.generated_questions, self.batch_rewards], feed_dict = feed)
		
		return loss, images, answers, questions, rewards

	def compute_mrr(self, preds, gt_indices, images, round_num, epoch):
		"""
		NOTE: BATCH SIZE HAS TO BE SMALL ~10 - 15 for this to hold in memory.
 		At each round we generate predictions from Q Bot across our batch.
		We then sort all the images in the validation set according to their distance to the
		given prediction and find the ranking of the true input image.
		===================================
		INPUTS:
		preds = float [batch_size, IMG_REP_DIM]
		gt_indices = float [batch_size] (indices of the ground truth images)
		images = float [VALIDATION_SIZE, IMG_REP_DIM]
		===================================
		OUTPUTS:
		"""
		validation_data_sz = tf.shape(images)[0]
		batch_data_sz = tf.shape(preds)[0]
		
		# Tile the predictions and images tensors to be of the same dimenions,
		# namely (Validation Data Size, Preds, Img Dimensions)
		preds_expanded = tf.tile(tf.expand_dims(preds, axis=0), tf.constant([validation_data_sz, 1, 1]))
		images_expanded = tf.tile(tf.expand_dims(images, axis=1), tf.constant([1, batch_data_sz, 1]))
		
		# Compute L2 distances.
		# Each column represents L2 distances between a predicted image and all val images.
		# Dim: (Preds, Validation Data Size)
		l2_distances = tf.transpose(tf.squeeze(tf.sqrt(tf.reduce_sum(tf.square(preds_expanded - images_expanded), axis=2)))) 
		
		# Sort the values in each row, i.e. sort all the similarity between all validation images
		# and predicted image.
		# Dim: (Preds, Validation Data Size)
		_, sorted_img_indices = tf.nn.top_k(
			tf.transpose(l2_distances), # (preds, validation data size)
			k=validation_data_sz,
			sorted=True,
		)

		# Unstack this matrix into a list of tensors
		# Each tensor in the list provides the indices of the validation images, in order from
		# farthest from the prediction, to closest to the prediction.
		sorted_img_indices_list = tf.unstack(sorted_img_indices)

		# Find the position of the image index corresponding to ground truth picture
		pos_gt = []
		for i, l in enumerate(sorted_img_indices_list):
			sorted_gt_pos = tf.argmax(tf.cast(tf.equal(l, gt_indices[i]), dtype=tf.int32), axis=0)
			pos_gt.append(sorted_gt_pos)
		percentage_rank_gt = (np.array(pos_gt) + 1) / validation_data_sz  # + 1 to account for 0 indexing
		return percentage_rank_gt

	def show_dialog(self, sess, image, caption, answer):
		feed = {
			self.images:images,
			self.captions:captions,
			self.caption_lengths: caption_lengths
			self.supervised_learning_rounds:0
		}
		images, answers, questions = sess.run([self.generated_images, self.generated_answers, self.generated_questions], feed_dict = feed)
		
		return loss, images, answers, questions, rewards


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