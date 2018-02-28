import tensorflow as tf

class config():
    class Q():
    #Vocabulary size for Q bot
        num_actions = 3
    # Possible guesses at the end
        num_classes = 144
    #epsilon for greediness
        
        epsilon_test = 0
        epsilon_start = 1
        epsilon_end = 0.1
        epsilon = epsilon_start
        #Number of iterations for decay
        iterations = 2000

    # discount factor
        gamma = 0.99
## Parameters for Abot
    class A():
    #Vocabulary size for A bot
        num_actions = 4
    #epsilon for greediness
        
        epsilon_test = 0
        epsilon_start = 1
        epsilon_end = 0.1
        epsilon = epsilon_start
        #Number of iterations for decay
        iterations = 50
    # discount factor
        gamma = 0.99

##Training parameters
    # Need Q bot to guess image representation at every round?
    guess_every_round = False
    # Number of rounds of dialog before cutoff
    max_dialog_rounds = 2
    # Batch Size
    batch_size = 150
    num_iterations = 20000
    verbose = False
## Evaluation parameters
    test_batch = 10
    eval_every = 50
## Smoothing parameters for plot
    win_length = 21
    polyorder = 3
##Output files
    output_dir = "./results"
    DATA_DIR = "../synthetic_data/"
    DATA_FILE = "synthetic_data.csv"
