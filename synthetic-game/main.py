import tensorflow as tf
import numpy as np
from model import Dialog_Bots
from config import config
def main():
	model=Dialog_Bots(config)
	model.train()
	model.generate_graphs()
	model.show_dialog(image, caption)
