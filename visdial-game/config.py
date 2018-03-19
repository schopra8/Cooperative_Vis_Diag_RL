import tensorflow as tf
import os

class Config():
    # Language Parametrs
    NUM_TRAINING_SAMPLES = 82783
    NUM_VALIDATION_SAMPLES = 40504
    NUM_EPOCHS = 400
    SL_EPOCHS = 15
    START_TOKEN_IDX = 8846
    END_TOKEN_IDX = 8847
    MAX_CAPTION_LENGTH = 40
    MAX_QUESTION_LENGTH = 22
    MAX_ANSWER_LENGTH = 22
    VOCAB_SIZE = 8845 + 2 + 1
    VGG_IMG_REP_DIM = 4096
    IMG_REP_DIM = 300
    EMBEDDING_SIZE = 300
    learning_rate = 1e-3
    batch_size = 40
    eval_batch_size = 15
    num_dialog_rounds = 10
    max_gradient_norm = 5
    model_save_directory = "../visdial_results/"
    best_save_directory = "../visdial_results/best"
    show_every = 10
    if not os.path.isdir(model_save_directory):
        os.makedirs(model_save_directory)
    if not os.path.isdir(best_save_directory):
        os.makedirs(best_save_directory)
    eval_every = 1000
    save_every = 1000
    #number of models to keep
    keep = 1
    class Q():
        gamma = 1
        hidden_dims = 512

    class A():
        gamma = 1
        hidden_dims = 512

