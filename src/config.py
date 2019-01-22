lr = 1e-3 # learning rate
n_epochs = 10000  # number of epochs
lamda = 0.001  # l2 loss regulation weight

batch_size = 1000
n_layers = 3 # number of layers
shape = (64, 32, 16) # network shape
# path settings
train_filename = "../data/ml-1m/train_users.dat"
test_filename = "../data/ml-1m/test_users.dat"
valid_filename = "../data/ml-1m/valid_users.dat"
