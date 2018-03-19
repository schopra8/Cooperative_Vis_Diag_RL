import tensorflow as tf
import numpy as np
from dataloader import DataLoader
from bots import DeepQBot
from bots import DeepABot
import math
import string
import pdb

class model():
    def __init__(self, config):
        """
        Sets up the configuration parameters, creates Q Bot + A Bot.
        """
        self.config = config
        self.embedding_matrix = tf.get_variable("word_embeddings", shape=[self.config.VOCAB_SIZE, self.config.EMBEDDING_SIZE])
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.Qbot = DeepQBot(self.config, self.embedding_matrix)
        self.Abot = DeepABot(self.config, self.embedding_matrix)
        self.add_placeholders()
        self.add_loss_op()
        self.add_update_op()
        self.best_model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.config.keep)

        # Files are expected to be in ../data
        # self.dataloader = DataLoader('visdial_params.json', 'visdial_data.h5',
                            # 'data_img.h5', ['train'])

    def add_placeholders(self):
        """
        Adds placeholders needed for training
        """
        self.vgg_images = tf.placeholder(tf.float32, shape=[None, self.config.VGG_IMG_REP_DIM])
        self.captions = tf.placeholder(tf.int32, shape=[None, self.config.MAX_CAPTION_LENGTH])
        self.caption_lengths = tf.placeholder(tf.int32, shape=[None])
        self.true_questions = tf.placeholder(tf.int32, shape=[None, self.config.num_dialog_rounds, self.config.MAX_QUESTION_LENGTH])
        self.true_answers = tf.placeholder(tf.int32, shape=[None, self.config.num_dialog_rounds, self.config.MAX_ANSWER_LENGTH])
        self.true_question_lengths = tf.placeholder(tf.int32, shape=[None, self.config.num_dialog_rounds])
        self.true_answer_lengths = tf.placeholder(tf.int32, shape=[None, self.config.num_dialog_rounds])
        self.num_supervised_learning_rounds = tf.placeholder(tf.int32, shape=[])
        self.generated_images = tf.placeholder(tf.float32, shape=[None, self.config.IMG_REP_DIM])
        self.gt_indices = tf.placeholder(tf.int32, shape=[])

    def add_loss_op(self):
        """
        Construct the loss op. 
        """
        self.loss, self.generated_questions, self.generated_answers, self.generated_images, self.batch_rewards = self.run_dialog_sl()
        self.loss_rl, self.generated_questions_rl, self.generated_answers_rl, self.generated_images_rl, self.batch_rewards_rl = self.run_dialog_rl()
        self.mrr = self.compute_mrr()
        avg_batch_rewards = tf.add_n(self.batch_rewards) / len(self.batch_rewards)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('avg_batch_rewards', avg_batch_rewards)

    def add_update_op(self):
        """
        Construct the update op for our networks.
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(self.config.learning_rate, self.global_step, 1000, 0.997592083))
        grads, variables = map(list, zip(*optimizer.compute_gradients(self.loss)))
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.max_gradient_norm)
        clipped_grads_vars = zip(clipped_grads, variables)
        self.update_op = optimizer.apply_gradients(clipped_grads_vars, global_step=self.global_step)

    def rl_run_dialog_round(self, i, Q_state, A_state, A_fact, prev_image_predictions):
        """
        Produce a round of dialog in a reinforcement learning setting, i.e. the questions produced by the Q-Bot are fed to the A-Bot and vice versa.
        """
        # Generate questions with Q-Bot
        question_logits, question_lengths, generated_questions = self.Qbot.get_questions(Q_state, supervised_training=False)

        # Update A-Bot state & generate answers
        encoded_questions = self.Abot.encode_questions(tf.nn.embedding_lookup(self.embedding_matrix, generated_questions), question_lengths)
        A_state, embedded_images = self.Abot.encode_state_histories(A_fact, self.vgg_images, encoded_questions, A_state)
        answer_logits, answer_lengths, generated_answers = self.Abot.get_answers(A_state, supervised_training=False)

        # Generate facts from dialogue and encode in both bots
        facts, fact_lengths = self.concatenate_q_a(generated_questions, question_lengths, generated_answers, answer_lengths)
        embedded_facts = tf.nn.embedding_lookup(self.embedding_matrix, facts)
        A_fact = self.Abot.encode_facts(embedded_facts, fact_lengths)
        Q_fact = self.Qbot.encode_facts(embedded_facts, fact_lengths)

        # Update Q-Bot state, guess the image, and compute rewards
        Q_state = self.Qbot.encode_state_histories(Q_fact, Q_state)
        generated_images = self.Qbot.generate_image_representations(Q_state)
        rewards = tf.reduce_sum(tf.square(prev_image_predictions - embedded_images), axis=1) - tf.reduce_sum(tf.square(generated_images - embedded_images), axis=1)
        batch_rewards = tf.reduce_mean(rewards)

        # Generate masks to zero out loss computed after stop tokens
        question_masks = 1 - tf.cast(tf.equal(generated_questions, tf.zeros(tf.shape(generated_questions), dtype=tf.int32)), tf.float32)
        answer_masks = 1-tf.cast(tf.equal(generated_answers, tf.zeros(tf.shape(generated_answers), dtype=tf.int32)), tf.float32)

        # Compute losses
        prob_questions = tf.reduce_max(tf.nn.softmax(question_logits), axis=2)
        prob_answers = tf.reduce_max(tf.nn.softmax(answer_logits), axis=2)
        negative_log_prob_exchange = -tf.reduce_sum(tf.log(prob_questions)*question_masks, axis=1) - tf.reduce_sum(tf.log(prob_answers)*answer_masks, axis=1)
        loss = tf.reduce_mean(negative_log_prob_exchange*rewards)
        return [loss, Q_state, A_state, A_fact, generated_questions, generated_answers, generated_images, batch_rewards]

    def sl_run_dialog_round(self, i, Q_state, A_state, A_fact):
        """
        Produce a round of dialog in a supervised learning setting, i.e. the Q-Bot is given ground truth questions and A-Bot is given ground truth answers.
        """
        # Extract the ground truth questions and answers (needed for supervised training)
        true_questions = self.true_questions[:,i,:]
        true_question_lengths = self.true_question_lengths[:,i]
        true_answers = self.true_answers[:,i,:]
        true_answer_lengths = self.true_answer_lengths[:,i]

        # Generate questions with Q-Bot
        question_logits, question_lengths, _ = self.Qbot.get_questions(
            Q_state,
            true_questions=true_questions,
            true_question_lengths=true_question_lengths,
            supervised_training=True
        )
        
        # Update A-Bot state & generate answers
        encoded_questions = self.Abot.encode_questions(tf.nn.embedding_lookup(self.embedding_matrix, true_questions), true_question_lengths)
        A_state, embedded_images = self.Abot.encode_state_histories(A_fact, self.vgg_images, encoded_questions, A_state)
        answer_logits, answer_lengths, _ = self.Abot.get_answers(
            A_state,
            true_answers=true_answers,
            true_answer_lengths=true_answer_lengths,
            supervised_training=True
        )

        # Generate and apply masks
        question_masks = 1-tf.cast(tf.equal(true_questions, tf.zeros(tf.shape(true_questions), dtype=tf.int32)), tf.float32)
        answer_masks = 1-tf.cast(tf.equal(true_answers, tf.zeros(tf.shape(true_answers), dtype=tf.int32)), tf.float32)
        true_questions = true_questions[:,:tf.shape(question_logits)[1]]
        question_masks = question_masks[:,:tf.shape(question_logits)[1]]
        true_answers = true_answers[:,:tf.shape(answer_logits)[1]]
        answer_masks = answer_masks[:,:tf.shape(answer_logits)[1]]

        # Generate facts from dialogue and encode in both bots
        facts, fact_lengths = self.concatenate_q_a(true_questions, question_lengths, true_answers, answer_lengths)
        embedded_facts = tf.nn.embedding_lookup(self.embedding_matrix, facts)
        A_fact = self.Abot.encode_facts(embedded_facts, fact_lengths)
        Q_fact = self.Qbot.encode_facts(embedded_facts, fact_lengths)

        # Update Q-Bot state and guess the image
        Q_state = self.Qbot.encode_state_histories(Q_fact, Q_state)
        image_guess = self.Qbot.generate_image_representations(Q_state)

        # Compute losses
        true_questions = tf.concat([true_questions[:,1:], tf.zeros([tf.shape(self.true_questions)[0],1], dtype=tf.int32)], axis=1)
        true_answers = tf.concat([true_answers[:,1:], tf.zeros([tf.shape(self.true_answers)[0],1], dtype=tf.int32)], axis=1)
        question_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=question_logits, labels=true_questions)*question_masks,
                axis=1)
            )
        answer_loss = tf.reduce_mean(
                tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=answer_logits, labels=true_answers)*answer_masks,
                axis=1)
        )
        image_loss = tf.reduce_mean(tf.nn.l2_loss(image_guess - embedded_images))
        loss = question_loss + answer_loss + image_loss
        return [loss, Q_state, A_state, A_fact, true_questions, true_answers, image_guess, tf.constant(0.0)]

    def run_dialog_sl(self):
        """
        Run dialog between two bots for a variable number of rounds with ground truths.
        """
        caption_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, self.captions)
        Q_state = self.Qbot.encode_captions(caption_embeddings, self.caption_lengths)
        A_state, _ = self.Abot.encode_images_captions(
            caption_embeddings,
            self.vgg_images,
            self.caption_lengths
        )
        A_fact = self.Abot.encode_facts(caption_embeddings, self.caption_lengths)
        _ = self.Qbot.generate_image_representations(Q_state)
        loss = 0.0
        generated_questions = []
        generated_answers = []
        generated_images = []
        cumulative_rewards = []

        for i in xrange(self.config.num_dialog_rounds):
            outputs = self.sl_run_dialog_round(i, Q_state, A_state, A_fact)
            round_loss, Q_state, A_state, A_fact, questions, answers, image_predictions, rewards = outputs
            loss += round_loss

            generated_questions.append(questions)
            generated_answers.append(answers)
            cumulative_rewards.append(rewards)
            generated_images.append(image_predictions)

        return loss, generated_questions, generated_answers, generated_images, cumulative_rewards

    def run_dialog_rl(self):
        """
        Run dialog between two bots for a variable number of rounds, with RL.
        """
        caption_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, self.captions)
        Q_state = self.Qbot.encode_captions(caption_embeddings, self.caption_lengths)
        A_state, _ = self.Abot.encode_images_captions(
            caption_embeddings,
            self.vgg_images,
            self.caption_lengths
        )
        A_fact = self.Abot.encode_facts(caption_embeddings, self.caption_lengths)
        prev_image_predictions = self.Qbot.generate_image_representations(Q_state)
        loss = 0.0
        generated_questions = []
        generated_answers = []
        generated_images = []
        cumulative_rewards = []

        for i in xrange(self.config.num_dialog_rounds):
            outputs = self.rl_run_dialog_round(i, Q_state, A_state, A_fact, prev_image_predictions)
            round_loss, Q_state, A_state, A_fact, questions, answers, image_predictions, rewards = outputs
            loss += round_loss

            generated_questions.append(questions)
            generated_answers.append(answers)
            cumulative_rewards.append(rewards)
            generated_images.append(image_predictions)
            prev_image_predictions = image_predictions

        return loss, generated_questions, generated_answers, generated_images, cumulative_rewards

    def train(self, sess, num_epochs, batch_size):
        """
        Train the Q-Bot and A-Bot first with supervised learning, then curriculum learning, and then completely with RL.
        """
        self.evaluate(sess, 0, True)
        summary_writer = tf.summary.FileWriter(self.config.model_save_directory, sess.graph)
        best_dev_loss = float('Inf')
        curriculum = 10
        for i in xrange(num_epochs):
            num_batches = self.config.NUM_TRAINING_SAMPLES / batch_size + 1
            progbar = tf.keras.utils.Progbar(target=num_batches)
            if i < self.config.SL_EPOCHS:
                curriculum = 10
            else:
                curriculum -= 1
            if curriculum < 0:
                curriculum = 0
            batch_generator = self.dataloader.getTrainBatch(batch_size)
            for j, batch in enumerate(batch_generator):
                loss, global_step = self.train_on_batch(sess, batch, summary_writer, supervised_learning_rounds=curriculum)
                prog_values = [("Loss", loss)]
                if global_step % self.config.eval_every == 0:
                    dev_loss, dev_MRR = self.evaluate(sess, i, True)
                    self.write_summary(dev_loss, "dev/loss_total", summary_writer, global_step)

                    print("DEV MRR: {}").format(dev_MRR)

                    #self.write_summary(tf.reduce_mean(dev_MRR), "dev/MRR_average", summary_writer, global_step)
                    if dev_loss < best_dev_loss:
                        print "New Best Model! Saving Best Model Weights!"
                        best_dev_loss = dev_loss
                        self.best_model_saver.save(sess, self.config.best_save_directory, global_step=global_step)
                        print "Done Saving Model Weights!"
                    prog_values.append((["Dev Loss", dev_loss]))
                progbar.update(j+1, prog_values)

                if global_step % self.config.save_every == 0:
                    print "Saving Model Weights!"
                    self.saver.save(sess, self.config.model_save_directory, global_step=global_step)
                    print "Done Saving Model Weights!"
                if global_step% self.config.show_every == 0:
                    self.show_dialog(sess, batch)

    def train_on_batch(self, sess, batch, summary_writer, supervised_learning_rounds=10):
        """ Train bots on a batch of data.
        """
        images, captions, caption_lengths, true_questions, true_question_lengths, true_answers, true_answer_lengths, _ = batch
        feed = {
            self.vgg_images: images,
            self.captions: captions,
            self.caption_lengths: caption_lengths,
            self.true_questions: true_questions,
            self.true_question_lengths: true_question_lengths,
            self.true_answers: true_answers,
            self.true_answer_lengths: true_answer_lengths,
            self.num_supervised_learning_rounds: supervised_learning_rounds
        }
        summary, _, global_step, loss = sess.run([self.summaries, self.update_op, self.global_step, self.loss], feed_dict=feed)
        summary_writer.add_summary(summary, global_step)
        self.write_summary(loss, 'train_loss', summary_writer, global_step)
        return loss, global_step

    def write_summary(self, value, tag, summary_writer, global_step):
        """
        Write a single summary value to tensorboard.
        """
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        summary_writer.add_summary(summary, global_step)

    def evaluate(self, sess, epoch, compute_MRR=False):
        """
        Evalaute the bots with validation data.
        """
        # files are expected to be in ../data
        eval_dataloader = DataLoader('visdial_params.json', 'visdial_data.h5',
                            'data_img.h5', ['val'])
        dev_loss = 0
        dev_mrr = tf.zeros(shape=[self.config.num_dialog_rounds])
        dev_batch_generator = eval_dataloader.getEvalBatch(self.config.eval_batch_size)
        num_batches = math.ceil(self.config.NUM_VALIDATION_SAMPLES / self.config.batch_size + 1)

        num_batches = 1

        progbar = tf.keras.utils.Progbar(target=num_batches)
        for i, batch in enumerate(dev_batch_generator):
            loss, _, _, _, _, mrr = self.eval_on_batch(sess, batch, compute_MRR)
            dev_loss += loss
            dev_mrr += tf.divide(tf.reduce_sum(mrr, axis=0), num_batches)
            if i == 0:
                break
            progbar.update(i+1)
        
        return dev_loss, dev_mrr

    def eval_on_batch(self, sess, batch, run_mrr):
        """
        Evaluate the bots on a batch of data.
        """
        images, captions, caption_lengths, true_questions, true_question_lengths, true_answers, true_answer_lengths, gt_indices = batch
        feed = {
            self.vgg_images: images,
            self.captions: captions,
            self.caption_lengths: caption_lengths,
            self.true_questions: true_questions,
            self.true_question_lengths: true_question_lengths,
            self.true_answers: true_answers,
            self.true_answer_lengths: true_answer_lengths,
            self.num_supervised_learning_rounds: 0,
            self.gt_indices: gt_indices,
        }
        loss, questions, answers, images, rewards = sess.run([self.loss, self.generated_questions, self.generated_answers, self.generated_images, self.batch_rewards], feed_dict = feed)

        if run_mrr:
            feed[self.generated_images] = images
            mrr = sess.run([self.mrr], feed_dict = feed)
        else:
            mrr = None
        return loss, images, answers, questions, rewards, mrr

    def compute_mrr(self):
        """
        At each round we generate predictions from Q Bot across our batch.
        We then sort all the images in the validation set according to their distance to the
        given prediction and find the ranking of the true input image.
        """
        validation_data_sz = self.config.NUM_VALIDATION_SAMPLES
        batch_data_sz = tf.shape(self.generated_images)[0]

        # TODO: Fetch all validation images
        embedded_ground_truth_images = tf.expand_dims(tf.expand_dims(tf.stack(self.Abot.convert_images(self.vgg_images)), axis=0), axis=3)
        embedded_ground_truth_images = tf.Print(embedded_ground_truth_images, [embedded_ground_truth_images, tf.shape(embedded_ground_truth_images)], summarize=15)

        # Tile the predictions and images tensors to be of the same dimenions,
        # namely (Validation Data Size, Num Rounds of Dialogs, Preds, Img Dimensions)
        preds_expanded = tf.tile(tf.expand_dims(self.generated_images, axis=0), tf.constant([validation_data_sz, 1, 1, 1]))
        images_expanded = tf.tile(embedded_ground_truth_images, tf.constant([1, self.config.num_dialog_rounds, 1, 1]))
        batch_tiling = tf.multiply(tf.constant([1, 1, 1, 1]) * batch_data_sz, tf.constant([0, 0, 0, 1]))
        images_expanded = tf.transpose(tf.tile(images_expanded, batch_tiling), perm=[0, 1, 3, 2])

        # Compute L2 distances.
        # Each column represents L2 distances between a predicted image and all val images.
        # Dim: (Preds, Num Rounds of Dialog, Validation Data Size)
        l2_distances = tf.transpose(tf.squeeze(tf.sqrt(tf.reduce_sum(tf.square(preds_expanded - images_expanded), axis=3))), perm=[2,1,0])

        # Sort the values in each row, i.e. sort all the similarity between all validation images
        # and predicted image.
        # Dim: (Preds, num rounds of dialog, Validation Data Size)
        _, sorted_img_indices = tf.nn.top_k(
            l2_distances, # (preds, num rounds of dialog, validation data size)
            k=validation_data_sz,
            sorted=True
        )
        print sorted_img_indices

        # Unstack this matrix into a list of tensors
        # Each tensor in the list provides the indices of the validation images, in order from
        # farthest from the prediction, to closest to the prediction.
        partitions = tf.range(self.config.eval_batch_size)
        partitioned_sorted_img_indices = tf.dynamic_partition(sorted_img_indices, partitions, self.config.eval_batch_size)
        sorted_img_indices_list = tf.unstack(partitioned_sorted_img_indices)
        print sorted_img_indices_list

        # Find the position of the image index corresponding to ground truth picture
        pos_gt = []
        for i, l in enumerate(sorted_img_indices_list):
            dialog_gt = []
            for j in xrange(self.config.num_dialog_rounds):
                sorted_gt_pos = tf.argmax(tf.cast(tf.equal(dialog_round, self.gt_indices[i]), dtype=tf.int32), axis=0)
                dialog_gt.append(sorted_gt_pos)
            pos_gt.append(dialog_gt)
        # TODO: is higher or lower better? right now 0 means perfect
        percentage_rank_gt = np.array(pos_gt) + 1 / float(validation_data_sz)  # + 1 to account for 0 indexing

        tf.Print(sorted_img_indices_list, [l2_distances, sorted_img_indices_list], "debugging", summarize=15)

        return percentage_rank_gt

    def show_dialog(self, sess, batch):
        """
        Show a sample dialog produced betweeb the Q-Bot and A-Bot.
        """
        images, captions, caption_lengths, true_questions, true_question_lengths, true_answers, true_answer_lengths, gt_indices = batch
        feed = {
            self.vgg_images: images,
            self.captions: captions,
            self.caption_lengths: caption_lengths,
            self.true_questions: true_questions,
            self.true_question_lengths: true_question_lengths,
            self.true_answers: true_answers,
            self.true_answer_lengths: true_answer_lengths,
            self.num_supervised_learning_rounds: 0
        }
        questions, answers, images, rewards = sess.run([self.generated_questions_rl, self.generated_answers_rl, self.generated_images_rl, self.batch_rewards_rl], feed_dict=feed)
        ind2word = self.dataloader.ind2word
        ind2word[0] = '<NONE>'
        questions = [np.vectorize(ind2word.__getitem__)(np.asarray(question)) for question in questions]
        answers = [np.vectorize(ind2word.__getitem__)(np.asarray(answer)) for answer in answers]
        captions = [np.vectorize(ind2word.__getitem__)(np.asarray(caption)) for caption in captions]
        i = 0
        for batch_idx, (image, caption) in enumerate(zip(images, captions)):
            print "Image Index: {}".format(gt_indices[i])
            print " ".join(caption)
            i += 1
            for round_idx, (question, answer) in enumerate(zip(questions,answers)):
                print "Round Number: %d" %round_idx
                print "Question:" + " ".join(question[batch_idx,:])
                print "Answer:" + " ".join(answer[batch_idx, :])

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
        max_size = self.config.MAX_QUESTION_LENGTH + self.config.MAX_ANSWER_LENGTH
        question_answer_pair_lengths = tf.add(question_lengths, answer_lengths)
        padded_question_answer_pairs = tf.Variable([], dtype=tf.int32, trainable=False)
        def body(i, padded_question_answer_pairs):
            stripped_question_answer_pair = tf.expand_dims(tf.concat([questions[i,0:question_lengths[i]],answers[i,0:answer_lengths[i]]], axis=0), axis=0)
            num_pad = max_size - question_answer_pair_lengths[i]
            paddings = tf.multiply(tf.constant([[1, 1],[1, 1]]),  num_pad)
            paddings = tf.multiply(paddings, tf.constant([[0, 0],[0, 1]]))
            padded_question_answer_pairs = tf.concat([padded_question_answer_pairs, tf.squeeze(tf.pad(stripped_question_answer_pair, paddings, "CONSTANT"))], axis=0)
            return tf.add(i, 1), padded_question_answer_pairs

        i = tf.constant(0)
        while_condition = lambda i, padded_question_answer_pairs: tf.less(i, tf.shape(questions)[0])
        _, padded_question_answer_pairs = tf.while_loop(
            while_condition,
            body,
            [i, padded_question_answer_pairs],
            parallel_iterations=1,
            shape_invariants=[i.get_shape(), tf.TensorShape([None])]
        )
        padded_question_answer_pairs = tf.reshape(padded_question_answer_pairs, [-1, max_size])
        return padded_question_answer_pairs, question_answer_pair_lengths
