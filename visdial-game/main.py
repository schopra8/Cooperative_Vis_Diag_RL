from config import Config
from model import model
import tensorflow as tf
def main():
    config = Config()
    visdial_bots = model(config)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(config.model_save_directory)
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
        if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
            print ("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            visdial_bots.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        visdial_bots.train(sess, num_epochs = config.num_epochs, batch_size = config.batch_size)


if __name__ == '__main__':
    main()
