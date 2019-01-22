from data import *
from DDN import DDN
import config
import os

data = Data(train_file=config.train_filename, valid_file=config.valid_filename, test_file=config.test_filename)

model = DDN(data, batch_size=config.batch_size, n_layers=config.n_layers)
model.train(config.n_epochs, lr=config.lr, optimizer='Adam')
model.predict(mode='valid')
model.predict(mode='test')
print(config.train_filename)
print(config.valid_filename)
print(config.test_filename)
print(config.batch_size)
print(config.shape)
