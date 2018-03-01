import tensorflow as tf

class config():
    class Q():
    #Vocabulary size for Q bot
        num_actions = 3
    # Possible guesses at the end
        num_classes = 144
    #epsilon for exploration
        epsilon_test = 0
        epsilon_start = 1
        epsilon_end = 0.3
        epsilon = epsilon_start
        #Number of iterations for decay
        iterations = 100000
        q_bot_lookup = {0: 'X', 1:'Y', 2:'Z'}
    # discount factor
        gamma = 1
## Parameters for Abot
    class A():
    #Vocabulary size for A bot
        num_actions = 4
    #epsilon for exploration
        epsilon_test = 0
        epsilon_start = 1
        epsilon_end = 0.3
        epsilon = epsilon_start
        #Number of iterations for decay
        iterations = 100000
    # discount factor
        gamma = 1

##Training parameters
    # caption lookup for reference
    caption_lookup = {0: [0,1], 1: [0,2], 2:[1,0], 3:[1,2], 4: [2,0], 5:[2,1]}
    # Need Q bot to guess image representation at every round?
    guess_every_round = False
    # Number of rounds of dialog before cutoff
    max_dialog_rounds = 2
    # Batch Size
    batch_size = 100
    num_iterations = 50000
    verbose = False
## Evaluation parameters
    test_batch = 50
    # test_batch = 1
    eval_every = 50
    show_dialog = False
    dialogs_to_show = 3
## Smoothing parameters for plot
    win_length = 21
    polyorder = 3
##Output files
    output_dir = "./results"
    DATA_DIR = "../synthetic_data/"
    DATA_FILE = "synthetic_data.csv"
    # DATA_FILE = "small.csv"
