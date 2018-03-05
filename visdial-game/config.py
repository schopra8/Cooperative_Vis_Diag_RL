import tensorflow as tf

class config():
    # Language Parametrs
    START_TOKEN = '<GO>'
    END_TOKEN = '<STOP>'
    MAX_QUESTION_LENGTH = 20
    MAX_ANSWER_LENGTH = 20
    VOCAB_SIZE = 1 # TODO: FIX THIS

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
