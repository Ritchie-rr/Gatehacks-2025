import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /opt/render/project/src/backend
PROJECT_ROOT = os.path.dirname(BASE_DIR)               # /opt/render/project/src

MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "best.pt")


'''
Bidirectional LSTM Parameters
'''
HIDDEN_DIM = 256

NUM_CLASSES = 6

LSTM_LAYERS = 2

DROPOUT_RATE = 0.2

'''
Training Parameters
'''

LR = .001

EPOCHS = 100