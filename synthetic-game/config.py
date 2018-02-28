import tensorflow as tf

class config():
    class Q():
    #Vocabulary size for Q bot
        num_actions = 3
    # Possible guesses at the end
        num_classes = 144
    #epsilon for exploration
        epsilon = 0.4
        # epsilon_start = 1
        # epsilon_end = 0.1
        # epsilon_step = 0.001
    # discount factor
        gamma = 1
## Parameters for Abot
    class A():
    #Vocabulary size for A bot
        num_actions = 4
    #epsilon for exploration
        epsilon = 0.4
        # epsilon_start = 1
        # epsilon_end = 0.1
        # epsilon_step = 0.001
    # discount factor
        gamma = 1

##Training parameters
    # Need Q bot to guess image representation at every round?
    guess_every_round = False
    # Number of rounds of dialog before cutoff
    max_dialog_rounds = 2
    # Batch Size
    batch_size = 20
    num_iterations = 500
##Output files
    output_dir = "./results/"
    DATA_DIR = "../synthetic_data/"
    DATA_FILE = "synthetic_data.csv"
