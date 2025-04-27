import cntk
from cntk.layers import *
from cntk.io.transforms import *
import numpy as np
from cntk import Trainer
from cntk.ops import *
from cntk.io import *
from cntk.learners import sgd
import pylab
import matplotlib as plt
import pickle

with open('db_stars.pkl', 'rb') as stars_pickle:
    STARS = pickle.load(stars_pickle)


features = STARS['Train']['Features'].astype(np.float32) / 256.0  # Исходные цифры
labels = STARS['Train']['Labels']  # Результирующие цифры


def conv(n):  # кодирование в one-hot-coding(вектроры 0 и 1)
    return np.array([1 if x == n else 0 for x in range(85)])


labels = np.array([conv(x) for x in labels], dtype=np.float32)

features = features.reshape(-1,1,300,300)

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.01)

input_var = input_variable((1,300,300))  # Входная переменная
label_var = input_variable(85)  # Выходная переменныя

model = Sequential([
    Convolution((3,3), 32, init=glorot_uniform(), pad=True, activation=relu),
    BatchNormalization(map_rank=1),
    MaxPooling((3,3), strides=(2,2)),
    #  Convolution((3,3), 32, init=glorot_uniform(), pad=True, activation=relu),
    #  BatchNormalization(map_rank=1),
    #  MaxPooling((3,3), strides=(2,2)),
    Convolution((2,2), 64, init=glorot_uniform(), pad=True, activation=relu),
    BatchNormalization(map_rank=1),
    MaxPooling((2,2), strides=(1,1)),
    #  Dense(5000, init=glorot_uniform(), activation=relu),
    Dense(1500, init=glorot_uniform(), activation=relu),
    Dense(400, init=glorot_uniform(), activation=relu),
    Dense(85, init=glorot_uniform(), activation=None)
])

z = model(input_var)

ce = cntk.cross_entropy_with_softmax(z, label_var)
pe = cntk.classification_error(z, label_var)

minibatch_size = 4
lr_per_minibatch = cntk.learning_rate_schedule(0.01, cntk.UnitType.minibatch)

pp = cntk.logging.ProgressPrinter()

learner = cntk.adagrad(z.parameters, lr=lr_per_minibatch)
trainer = cntk.Trainer(z, (ce, pe), [learner], [pp])

cntk.logging.log_number_of_parameters(z)
progress = []

for epoch in range(15):
    perm = np.random.permutation(len(labels_train))
    taccuracy = 0; tloss = 0; cnt = 0
    for i in range(0, len(labels_train), minibatch_size):
        max_n = min(i + minibatch_size, len(labels_train))
        x = features_train[perm[i:max_n]]
        t = labels_train[perm[i:max_n]]
        trainer.train_minibatch({input_var:x, label_var:t})
        tloss += trainer.previous_minibatch_loss_average * trainer.previous_minibatch_sample_count
        taccuracy += trainer.previous_minibatch_evaluation_average * trainer.previous_minibatch_sample_count
        cnt += trainer.previous_minibatch_sample_count
        pp.update_with_trainer(trainer, with_metric=True)
    progress.append([tloss, taccuracy])
    pp.epoch_summary(with_metric=True)

vloss = np.average(ce.eval({input_var:features_test, label_var:labels_test}))
verr = np.average(pe.eval({input_var:features_test, label_var:labels_test}))
print("Loss={}, Error={}".format(vloss, verr))

z.save("star_nn.model")
