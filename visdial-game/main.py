from config import Config
from model import model
import tensorflow as tf
from dataloader import DataLoader

def main():
    config = Config()
    visdial_bots = model(config)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(config.model_save_directory)
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
        if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
            print "Reading model parameters from %s" % ckpt.model_checkpoint_path
            visdial_bots.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        visdial_bots.train(sess, num_epochs = config.NUM_EPOCHS, batch_size = config.batch_size)

        # eval_dataloader = DataLoader('visdial_params.json', 'visdial_data.h5',
        #                     'data_img.h5', ['val'])
        # dev_batch_generator = eval_dataloader.getEvalBatch(5)
        # batch = dev_batch_generator.next()
        # images, captions, caption_lengths, true_questions, true_question_lengths, true_answers, true_answer_lengths, gt_indices = batch
        # visdial_bots.show_dialog(sess, images, captions, caption_lengths, gt_indices)
        # visdial_bots.evaluate(sess, epoch=10, compute_MRR = True)

if __name__ == '__main__':
    main()
