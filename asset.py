import torch.nn as nn
import torch.optim as optim
from model import *
from pretreatment import *

preprocess_set={
    'bow':BOW,
    'tfidf':TFIDF,
    'ngram':NGram,
    'glove':GLOVE,
    'word2vec':WORD2VEC,
    'fasttext':FASTTEXT
}

model_set={
    'logistic':Logistic_OVO,
    'softmax':SoftmaxRegression,
    'rnn':nn.RNN,
    }

activate_set = {
    'relu':nn.ReLU,
    'tanh':nn.Tanh,
    'sigmoid':nn.Sigmoid,
    'leaky_relu':nn.LeakyReLU,
    'elu':nn.ELU,
    'selu':nn.SELU,
    'softmax':nn.Softmax,
    'none':nn.Identity
}

loss_set = {
    'ce':nn.CrossEntropyLoss,
    'bce':nn.BCEWithLogitsLoss,
    'mse':nn.MSELoss,
    'mae':nn.L1Loss
}

optim_set = {
    'adam':optim.Adam,
    'sgd':optim.SGD,
    'adamw':optim.AdamW,
    'rmsprop':optim.RMSprop,
    'adadelta':optim.Adadelta,
    'adagrad':optim.Adagrad,
    'adamax':optim.Adamax
}
