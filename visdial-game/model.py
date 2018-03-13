import tensorflow as tf
import numpy as np
from dataloader import DataLoader
from bots import DeepQBot
from bots import DeepABot


class model():
    def __init__(self, config):
        """ Sets up the configuration parameters, creates Q Bot + A Bot.
        """
        self.config = config
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
        self.add_loss_op()
        self.add_update_op()
        self.best_model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.config.keep)

        # files are expected to be in ../data
        self.dataloader = DataLoader('visdial_params.json', 'visdial_data.h5',
                            'data_img.h5', ['train'])

    def add_loss_op(self):
        self.loss, self.generated_questions, self.generated_answers, self.generated_images, self.batch_rewards = self.run_dialog()
        avg_batch_rewards = tf.add_n(self.batch_rewards) / len(self.batch_rewards)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('avg_batch_rewards', avg_batch_rewards)

    def add_update_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate = self.config.learning_rate)
        grads_vars = optimizer.compute_gradients(self.loss)
        grads, variables = map(list,zip(*optimizer.compute_gradients(self.loss)))
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.max_gradient_norm)
        clipped_grads_vars = zip(clipped_grads, variables)
        self.update_op = optimizer.apply_gradients(clipped_grads_vars, global_step = self.global_step)

    def add_placeholders(self):
        """
        Adds placeholders needed for training
        """
        self.images = tf.placeholder(tf.float32, shape = [None, self.config.VGG_IMG_REP_DIM])
        self.captions = tf.placeholder(tf.int32, shape = [None, self.config.MAX_CAPTION_LENGTH])
        self.caption_lengths = tf.placeholder(tf.int32, shape = [None])
        self.true_questions = tf.placeholder(tf.int32, shape = [None, self.config.num_dialog_rounds, self.config.MAX_QUESTION_LENGTH])
        self.true_answers = tf.placeholder(tf.int32, shape = [None, self.config.num_dialog_rounds, self.config.MAX_ANSWER_LENGTH])
        self.true_question_lengths = tf.placeholder(tf.int32, shape = [None, self.config.num_dialog_rounds])
        self.true_answer_lengths = tf.placeholder(tf.int32, shape = [None, self.config.num_dialog_rounds])
        self.supervised_learning_rounds = tf.placeholder(tf.int32, shape=[])

    
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
        captions = tf.nn.embedding_lookup(self.embedding_matrix, self.captions)
        Q_state = self.Qbot.encode_captions(captions, self.caption_lengths)
        A_state, embedded_images = self.Abot.encode_images_captions(captions, self.images, self.caption_lengths)
        A_fact = self.Abot.encode_facts(captions, self.caption_lengths)
        prev_image_guess = self.Qbot.generate_image_representations(Q_state)
        image_loss = 0
        loss = 0
        generated_questions = []
        generated_answers = []
        generated_images = []
        batch_rewards = []

        def rl_run_dialog_round(Q_state, A_state, A_fact, prev_image_guess, embedded_images):
            #Q-Bot generates question logits
            question_logits, question_lengths = self.Qbot.get_questions(Q_state, supervised_training = False)
            #Find embeddings of questions
            generated_questions = tf.argmax(question_logits, axis=2, output_type=tf.int32)
            #A-bot encodes questions
            encoded_questions = self.Abot.encode_questions(tf.nn.embedding_lookup(self.embedding_matrix, generated_questions))
            #A-bot updates state
            A_state = self.Abot.encode_state_histories(A_fact, embedded_images, encoded_questions, A_state)
            #Abot generates answer logits
            answer_logits, answer_lengths = self.Abot.get_answers(A_state, supervised_training=False)
            #Generate facts for that round of dialog
            generated_answers = tf.argmax(answer_logits, axis=2, output_type=tf.int32)
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
            rewards = tf.reduce_sum(tf.square(prev_image_guess - embedded_images), axis = 1) - tf.reduce_sum(tf.square(generated_images - embedded_images), axis = 1)
            batch_rewards = tf.reduce_mean(rewards)

            #### CHANGE HERE FOR UPDATING ONLY SINGLE BOT
            prob_questions = tf.reduce_max(tf.nn.softmax(question_logits), axis = 2)
            prob_answers = tf.reduce_max(tf.nn.softmax(answer_logits), axis = 2)
            negative_log_prob_exchange = -tf.reduce_sum(tf.log(prob_questions), axis=1) - tf.reduce_sum(tf.log(prob_answers), axis=1)
            loss = tf.reduce_sum(negative_log_prob_exchange*rewards)

            return [loss, Q_state, A_state, A_fact, generated_questions, generated_answers, generated_images, batch_rewards]

        def sl_run_dialog_round(Q_state, A_state, A_fact, embedded_images):
            ## ACCESS TRUE QUESTIONS  AND ANSWERS FOR THIS ROUND OF DIALOG
            questions = self.true_questions[:,i,:]
            answers = self.true_answers[:,i,:]
            true_question_lengths_round = self.true_question_lengths[:,i]
            true_answer_lengths_round = self.true_answer_lengths[:,i]
            #Generate questions based on current state
            question_outputs, question_lengths = self.Qbot.get_questions(
                Q_state,
                true_questions=questions,
                true_question_lengths=true_question_lengths_round,
                supervised_training=True
            )
            #Encode the true questions
            encoded_questions = self.Abot.encode_questions(tf.nn.embedding_lookup(self.embedding_matrix, questions))
            #Update A state based on true question
            A_state = self.Abot.encode_state_histories(A_fact, embedded_images, encoded_questions, A_state)
            # ABot Generates answers based on current state
            answer_outputs, answer_lengths = self.Abot.get_answers(
                A_state,
                true_answers=answers,
                true_answer_lengths=true_answer_lengths_round,
                supervised_training=True
            )
            #Generate facts from true questions and answers
            # question_outputs = tf.Print(question_outputs, [tf.shape(questions), tf.shape(question_outputs), tf.shape(answers), answer_outputs], "SHAPES++++++++++=====")
            questions = questions[:,:tf.shape(question_outputs)[1]]
            answers = answers[:,:tf.shape(answer_outputs)[1]]
            facts, fact_lengths = self.concatenate_q_a(questions, true_question_lengths_round, answers, true_answer_lengths_round)
            embedded_facts = tf.nn.embedding_lookup(self.embedding_matrix, facts)
            A_fact = self.Abot.encode_facts(embedded_facts, fact_lengths)
            Q_fact = self.Qbot.encode_facts(embedded_facts, fact_lengths)
            #Update state histories using current facts
            Q_state = self.Qbot.encode_state_histories(Q_fact, Q_state)
            #Guess image
            image_guess = self.Qbot.generate_image_representations(Q_state)
            #### Loss for supervised training
            # question_logits, question_order = tf.transpose(question_outputs), tf.transpose(question_outputs[1], perm=[1, 0])
            # answer_logits, answer_order = tf.transpose(answer_outputs[0], perm=[1, 0, 2]), tf.transpose(answer_outputs[1], perm=[1, 0])
            dialog_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = question_outputs, labels = questions))
            dialog_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = answer_outputs, labels = answers))
            image_loss = tf.reduce_sum(tf.nn.l2_loss(image_guess - embedded_images))
            loss = dialog_loss + image_loss

            return [loss, Q_state, A_state, A_fact, questions, answers, image_guess, tf.constant(0.0)]

        for i in xrange(self.config.num_dialog_rounds):
            x = tf.constant(i)
            outputs = tf.cond(
                tf.greater_equal(x, self.supervised_learning_rounds),
                lambda: rl_run_dialog_round(Q_state, A_state, A_fact, prev_image_guess, embedded_images),
                lambda: sl_run_dialog_round(Q_state, A_state, A_fact, embedded_images)
            )
            loss += outputs[0]
            _, Q_state, A_state, A_fact, questions, answers, prev_image_guess, new_batch_rewards = outputs
            generated_questions.append(questions)
            generated_answers.append(answers)
            batch_rewards.append(new_batch_rewards)
            generated_images.append(prev_image_guess)

        return loss, generated_questions, generated_answers, generated_images, batch_rewards

    def train(self, sess, num_epochs=400, batch_size=20):
        summary_writer = tf.summary.FileWriter(self.config.model_save_directory, sess.graph)
        best_dev_loss = float('Inf')
        curriculum = 0
        for i in xrange(num_epochs):
            num_batches = self.config.NUM_TRAINING_SAMPLES / batch_size + 1
            progbar = tf.keras.utils.Progbar(target=num_batches)
            if i<self.config.SL_EPOCHS:
                curriculum = 10
            else:
                curriculum -= 1
            if curriculum <0:
                curriculum = 0
            batch_generator = self.dataloader.getTrainBatch(self.config.batch_size)
            for j, batch in enumerate(batch_generator):
                loss, global_step = self.train_on_batch(sess, batch, summary_writer, supervised_learning_rounds = curriculum)
                prog_values=[("Loss", loss)]
                if global_step % self.config.eval_every == 0:
                    dev_loss, dev_MRR = self.evaluate(sess, i)
                    self.write_summary(dev_loss, "dev/loss_total", summary_writer, global_step)
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

  
    def train_on_batch(self, sess, batch, summary_writer, supervised_learning_rounds = 10):
        images, captions, caption_lengths, true_questions, true_question_lengths, true_answers, true_answer_lengths, _ = batch
        feed = {
            self.images:images,
            self.captions:captions,
            self.caption_lengths:caption_lengths,
            self.true_questions:true_questions,
            self.true_question_lengths:true_question_lengths,
            self.true_answers:true_answers,
            self.true_answer_lengths:true_answer_lengths,
            self.supervised_learning_rounds:supervised_learning_rounds,
        }
        summary, _, global_step, loss = sess.run([self.summaries, self.update_op, self.global_step, self.loss], feed_dict = feed)
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
        num_batches = self.config.NUM_VALIDATION_SAMPLES / self.config.batch_size + 1
        progbar = tf.keras.utils.Progbar(target=num_batches)
        for i, batch in enumerate(dev_batch_generator):
            true_images, _, _, _, _, _, _, gt_indices = batch
            loss, preds, _, _, _ = self.eval_on_batch(sess, batch)
            dev_loss += loss
            MRR = np.zeros([self.config.num_dialog_rounds])
            if compute_MRR:
                for round_number, p in enumerate(preds):
                    percentage_rank_gt = self.compute_mrr(p, gt_indices, true_images, round_number, epoch)
                    MRR[round_number] += tf.reduce_mean(percentage_rank_gt)
            progbar.update(i+1)
        return dev_loss, MRR

    def eval_on_batch(self, sess, batch):
        images, captions, caption_lengths, true_questions, true_question_lengths, true_answers, true_answer_lengths, _ = batch
        feed = {
            self.images:images,
            self.captions:captions,
            self.caption_lengths:caption_lengths,
            self.true_questions:true_questions,
            self.true_question_lengths:true_question_lengths,
            self.true_answers:true_answers,
            self.true_answer_lengths:true_answer_lengths,
            self.supervised_learning_rounds:0,
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

    def show_dialog(self, sess, images, captions, caption_lengths, answers):
        pass
        # feed = {
        #     self.images:images,
        #     self.captions:captions,
        #     self.caption_lengths: caption_lengths,
        # }
        # loss, generated_questions, generated_answers, generated_images, batch_rewards = self.run_dialog(0)
        # preds, gen_answers, gen_questions = sess.run([generated_images, generated_answers, generated_questions], feed_dict=feed)
        # return preds, gen_answers, gen_questions

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