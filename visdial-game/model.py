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
        """ Sets up the configuration parameters, creates Q Bot + A Bot.
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

        # files are expected to be in ../data
        self.dataloader = DataLoader('visdial_params.json', 'visdial_data.h5',
                            'data_img.h5', ['train'])

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

    def add_loss_op(self):
        self.loss, self.generated_questions, self.generated_answers, self.generated_images, self.batch_rewards = self.run_dialog()
        avg_batch_rewards = tf.add_n(self.batch_rewards) / len(self.batch_rewards)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('avg_batch_rewards', avg_batch_rewards)

    def add_update_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        grads, variables = map(list, zip(*optimizer.compute_gradients(self.loss)))
        # clip global_norm or norm??
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.max_gradient_norm)
        clipped_grads_vars = zip(clipped_grads, variables)
        self.update_op = optimizer.apply_gradients(clipped_grads_vars, global_step=self.global_step)

    def rl_run_dialog_round(self, i, Q_state, A_state, A_fact, prev_image_predictions):
        #Q-Bot generates question logits
        question_logits, question_lengths, generated_questions = self.Qbot.get_questions(Q_state, supervised_training=False)
        generated_questions = tf.Print(generated_questions, [tf.shape(generated_questions), generated_questions], "Generated Questions Shape & Tensor")
        #Find embeddings of questions
        question_masks = tf.cast(tf.equal(generated_questions, tf.zeros(tf.shape(generated_questions), dtype=tf.int32)), tf.float32)
        #A-bot encodes questions
        encoded_questions = self.Abot.encode_questions(tf.nn.embedding_lookup(self.embedding_matrix, generated_questions), question_lengths)
        #A-bot updates state
        A_state, embedded_images = self.Abot.encode_state_histories(A_fact, self.vgg_images, encoded_questions, A_state)
        #Abot generates answer logits
        answer_logits, answer_lengths, generated_answers = self.Abot.get_answers(A_state, supervised_training=False)
        #Generate facts for that round of dialog
        # question_lengths = tf.Print(question_lengths, [generated_questions, question_lengths, generated_answers, answer_lengths], "DEBUGGING:")
        answer_masks = 1-tf.cast(tf.equal(generated_answers, tf.zeros(tf.shape(generated_answers), dtype=tf.int32)), tf.float32)
        facts, fact_lengths = self.concatenate_q_a(generated_questions, question_lengths, generated_answers, answer_lengths)
        #Embed the facts into word vector space
        facts = tf.nn.embedding_lookup(self.embedding_matrix, facts)
        #Encode facts into both bots
        A_fact = self.Abot.encode_facts(facts, fact_lengths)
        Q_fact = self.Qbot.encode_facts(facts, fact_lengths)
        Q_state = self.Qbot.encode_state_histories(Q_fact, Q_state)
        # QBot Generates guess of image
        generated_images = self.Qbot.generate_image_representations(Q_state)
        #Calculate loss for this round
        rewards = tf.reduce_sum(tf.square(prev_image_predictions - embedded_images), axis=1) - tf.reduce_sum(tf.square(generated_images - embedded_images), axis=1)
        batch_rewards = tf.reduce_mean(rewards)

        #### CHANGE HERE FOR UPDATING ONLY SINGLE BOT
        prob_questions = tf.reduce_max(tf.nn.softmax(question_logits), axis=2)
        prob_answers = tf.reduce_max(tf.nn.softmax(answer_logits), axis=2)
        negative_log_prob_exchange = -tf.reduce_sum(tf.log(prob_questions)*question_masks, axis=1) - tf.reduce_sum(tf.log(prob_answers)*answer_masks, axis=1)
        loss = tf.reduce_mean(negative_log_prob_exchange*rewards)

        return [loss, Q_state, A_state, A_fact, generated_questions, generated_answers, generated_images, batch_rewards]

    def sl_run_dialog_round(self, i, Q_state, A_state, A_fact):
        ## ACCESS TRUE QUESTIONS AND ANSWERS FOR THIS ROUND OF DIALOG
        true_questions = self.true_questions[:,i,:]
        true_question_lengths = self.true_question_lengths[:,i]
        true_answers = self.true_answers[:,i,:]
        true_answer_lengths = self.true_answer_lengths[:,i]
        #Generate questions based on current state
        question_masks = 1-tf.cast(tf.equal(true_questions, tf.zeros(tf.shape(true_questions), dtype=tf.int32)), tf.float32)
        answer_masks = 1-tf.cast(tf.equal(true_answers, tf.zeros(tf.shape(true_answers), dtype=tf.int32)), tf.float32)
        question_logits, question_lengths, questions = self.Qbot.get_questions(
            Q_state,
            true_questions=true_questions,
            true_question_lengths=true_question_lengths,
            supervised_training=True
        )

        true_questions = tf.Print(true_questions, [tf.shape(true_questions), true_questions], "True Questions Shape & Tensor")
        true_question_lengths = tf.Print(true_question_lengths, [true_question_lengths], "True Question Lengths")
        question_masks = tf.Print(question_masks, [tf.shape(question_masks), question_masks], "Question Masks Shape & Tensor")

        questions = tf.Print(questions, [tf.shape(questions), questions], "Generated Questions Shape & Tensor")
        question_lengths = tf.Print(question_lengths, [question_lengths], "Generated Question Lengths")
        question_logits = tf.Print(question_logits, [tf.shape(question_logits), question_logits], "Question Logits Shape & Tensor")


        #Encode the true questions
        # question_lengths = tf.Print(question_lengths,[question_lengths, true_question_lengths], "Length of Questions:")
        encoded_questions = self.Abot.encode_questions(tf.nn.embedding_lookup(self.embedding_matrix, true_questions), true_question_lengths)
        # encoded_questions = tf.Print(encoded_questions, [encoded_questions], "Encoded Questions")
        #Update A state based on true question
        A_state, embedded_images = self.Abot.encode_state_histories(A_fact, self.vgg_images, encoded_questions, A_state)

        # A_state = tf.Print(A_state, ["A State", A_state])
        # embedded_images = tf.Print(embedded_images, [embedded_images], "Embedded Images")


        # ABot Generates answers based on current state
        answer_logits, answer_lengths, answers = self.Abot.get_answers(
            A_state,
            true_answers=true_answers,
            true_answer_lengths=true_answer_lengths,
            supervised_training=True
        )

        # answers = tf.Print(answers, [tf.shape(answers), answers], "Generated Answer Shape & Tensor")
        # answer_lengths = tf.Print(answer_lengths, [answer_lengths], "Generated Answer Lengths")

        #Generate facts from true questions and answers
        # question_logits = question_logits = tf.Print(question_logits, [tf.shape(questions), tf.shape(question_logits), tf.shape(answers), answer_logits], "SHAPES++++++++++=====")
        questions = true_questions[:,:tf.shape(question_logits)[1]]
        question_masks = question_masks[:,:tf.shape(question_logits)[1]]
        answers = true_answers[:,:tf.shape(answer_logits)[1]]
        answer_masks = answer_masks[:,:tf.shape(answer_logits)[1]]

        facts, fact_lengths = self.concatenate_q_a(questions, true_question_lengths, answers, true_answer_lengths)

        # facts = tf.Print(facts, [tf.shape(facts), facts], "facts")

        embedded_facts = tf.nn.embedding_lookup(self.embedding_matrix, facts)

        # embedded_facts = tf.Print(embedded_facts, [embedded_facts], "Embedded Facts")

        A_fact = self.Abot.encode_facts(embedded_facts, fact_lengths)

        # A_fact = tf.Print(A_fact, [A_fact], "A Fact")

        Q_fact = self.Qbot.encode_facts(embedded_facts, fact_lengths)

        # Q_fact = tf.Print(Q_fact, [Q_fact], "Q Fact")

        #Update state histories using current facts
        Q_state = self.Qbot.encode_state_histories(Q_fact, Q_state)

        # Q_state = tf.Print(Q_state, [Q_state], "Q State")

        #Guess image
        image_guess = self.Qbot.generate_image_representations(Q_state)
        #### Loss for supervised training
        # question_logits, question_order = tf.transpose(question_logits), tf.transpose(question_logits[1], perm=[1, 0])
        # answer_logits, answer_order = tf.transpose(answer_logits[0], perm=[1, 0, 2]), tf.transpose(answer_logits[1], perm=[1, 0])
        question_loss = tf.reduce_mean(
            tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=question_logits, labels=questions)*question_masks, axis=1))
        answer_loss = tf.reduce_mean(
                tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=answer_logits, labels=answers)*answer_masks, axis=1))
        image_loss = tf.reduce_mean(tf.nn.l2_loss(image_guess - embedded_images))

        loss = question_loss + answer_loss + image_loss
        return [loss, Q_state, A_state, A_fact, questions, answers, image_guess, tf.constant(0.0)]

    def run_dialog(self):
        """
        Function that runs dialog between two bots for a variable number of rounds, with a subset of rounds with supervised training
        ================================
        INPUTS:
        ================================
        OUTPUTS:
        """
        caption_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, self.captions)
        Q_state = self.Qbot.encode_captions(caption_embeddings, self.caption_lengths)
        A_state, reduced_dim_image_embeddings = self.Abot.encode_images_captions(
            caption_embeddings,
            self.vgg_images,
            self.caption_lengths
        )
        A_fact = self.Abot.encode_facts(caption_embeddings, self.caption_lengths)
        prev_image_predictions = self.Qbot.generate_image_representations(Q_state)
        image_loss = 0.0
        loss = 0.0
        generated_questions = []
        generated_answers = []
        generated_images = []
        cumulative_rewards = []

        for i in xrange(self.config.num_dialog_rounds):
            x = tf.constant(i)
            loss = tf.Print(loss, [x], "dialog_number: ")
            outputs = self.sl_run_dialog_round(i, Q_state, A_state, A_fact)
            # outputs = tf.cond(
            #     tf.greater_equal(x, self.num_supervised_learning_rounds),
            #     lambda: self.rl_run_dialog_round(i, Q_state, A_state, A_fact, prev_image_predictions),
            #     lambda: self.sl_run_dialog_round(i, Q_state, A_state, A_fact)
            # )
            round_loss, Q_state, A_state, A_fact, questions, answers, image_predictions, rewards = outputs
            loss += round_loss

            generated_questions.append(questions)
            generated_answers.append(answers)
            cumulative_rewards.append(rewards)
            generated_images.append(image_predictions)
            prev_image_predictions = image_predictions

        return loss, generated_questions, generated_answers, generated_images, cumulative_rewards

    def train(self, sess, num_epochs, batch_size):
        summary_writer = tf.summary.FileWriter(self.config.model_save_directory, sess.graph)
        best_dev_loss = float('Inf')
        curriculum = 10
        for i in xrange(num_epochs):
            num_batches = self.config.NUM_TRAINING_SAMPLES / batch_size + 1
            progbar = tf.keras.utils.Progbar(target=num_batches)
            if i<self.config.SL_EPOCHS:
                curriculum = 10
            else:
                curriculum -= 1
            if curriculum <0:
                curriculum = 0
            batch_generator = self.dataloader.getTrainBatch(batch_size)
            for j, batch in enumerate(batch_generator):
                loss, global_step = self.train_on_batch(sess, batch, summary_writer, supervised_learning_rounds=curriculum)
                prog_values = [("Loss", loss)]
                if global_step % self.config.eval_every == 0:
                    dev_loss, dev_MRR = self.evaluate(sess, i)
                    self.write_summary(dev_loss, "dev/loss_total", summary_writer, global_step)

                    print("DEV MRR: {}").format(dev_MRR)

                    self.write_summary(tf.reduce_mean(dev_MRR), "dev/MRR_average", summary_writer, global_step)
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
                    images, captions, caption_lengths, _, _, _, _, gt_indices = batch
                    self.show_dialog(sess, images, captions, caption_lengths, gt_indices)

    def train_on_batch(self, sess, batch, summary_writer, supervised_learning_rounds=10):
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
        """ Write a single summary value to tensorboard
        """
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        summary_writer.add_summary(summary, global_step)

    def evaluate(self, sess, epoch, compute_MRR = False):
        # files are expected to be in ../data
        eval_dataloader = DataLoader('visdial_params.json', 'visdial_data.h5',
                            'data_img.h5', ['val'])
        dev_loss = 0
        dev_batch_generator = eval_dataloader.getEvalBatch(self.config.batch_size)
        num_batches = math.ceil(self.config.NUM_VALIDATION_SAMPLES / self.config.batch_size + 1)
        progbar = tf.keras.utils.Progbar(target=num_batches)
        for i, batch in enumerate(dev_batch_generator):
            true_images, _, _, _, _, _, _, gt_indices = batch
            loss, preds, _, _, _ = self.eval_on_batch(sess, batch)
            dev_loss += loss
            MRR = np.zeros([self.config.num_dialog_rounds])
            if compute_MRR:
                for round_number, p in enumerate(preds):
                    percentage_rank_gt = self.compute_mrr(p, gt_indices, true_images, round_number, epoch)
                    MRR[round_number] += tf.divide(tf.reduce_mean(percentage_rank_gt), tf.constant(num_batches))
            progbar.update(i+1)
        return dev_loss, MRR

    def eval_on_batch(self, sess, batch):
        images, captions, caption_lengths, true_questions, true_question_lengths, true_answers, true_answer_lengths, _ = batch
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
        loss, questions, answers, images, rewards = sess.run([self.loss, self.generated_questions, self.generated_answers, self.generated_images, self.batch_rewards], feed_dict = feed)

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

    def show_dialog(self, sess, images, captions, caption_lengths, gt_indices):
        feed = {
            self.vgg_images: images,
            self.captions: captions,
            self.caption_lengths: caption_lengths,
            self.true_questions: np.zeros((1, self.config.num_dialog_rounds, self.config.MAX_QUESTION_LENGTH), dtype=np.int32),
            self.true_question_lengths: np.zeros((1, self.config.num_dialog_rounds), dtype=np.int32),
            self.true_answers: np.zeros((1, self.config.num_dialog_rounds, self.config.MAX_ANSWER_LENGTH), dtype=np.int32),
            self.true_answer_lengths: np.zeros((1, self.config.num_dialog_rounds), dtype=np.int32),
            self.num_supervised_learning_rounds: 0
        }

        questions, answers, images, rewards = sess.run([self.generated_questions, self.generated_answers, self.generated_images, self.batch_rewards], feed_dict = feed)
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
