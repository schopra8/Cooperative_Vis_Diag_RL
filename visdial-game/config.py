import tensorflow as tf

class Config():
    # Language Parametrs
    START_TOKEN = '<GO>'
    END_TOKEN = '<STOP>'
    START_TOKEN_IDX = 8846
    END_TOKEN_IDX = 8847
    MAX_QUESTION_LENGTH = 20
    MAX_ANSWER_LENGTH = 20
    VOCAB_SIZE = 8845 + 2
    IMG_REP_DIM = 300
    EMBEDDING_SIZE = 300
    batch_size = 40
    number_of_dialog_rounds = 10
    max_gradient_norm = 5

    class Q():
        gamma = 1
        hidden_dims = 512

    class A():
        gamma = 1
        hidden_dims = 512

    class FLAGS():
        train_dir = ''

