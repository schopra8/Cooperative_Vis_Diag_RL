import tensorflow as tf

class config():
    # Language Parametrs
    START_TOKEN = '<GO>'
    END_TOKEN = '<STOP>'
    START_TOKEN_IDX = -1 # TODO: FIX
    END_TOKEN_IDX = -1 # TODO: FIX
    MAX_QUESTION_LENGTH = 20
    MAX_ANSWER_LENGTH = 20
    VOCAB_SIZE = 1 # TODO: INCLUDING START and END TOKENS
    IMG_REP_DIM = 300 # TODO: FIX THIS
    EMBEDDING_SIZE = 300

    # Training Parameters

    class Q():
        gamma = 1
        hidden_dims = 512

    class A():
        gamma = 1
        hidden_dims = 512

##Output files
    output_dir = "./results"
    DATA_DIR = "../synthetic_data/"
    DATA_FILE = "synthetic_data.csv"
