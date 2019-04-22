import pickle

from utils import *

f = open('histories/cats_dogs_VGG.hist', "rb")
history = pickle.load(f)


plot_history(history)