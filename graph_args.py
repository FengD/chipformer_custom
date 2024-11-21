### CONFIGS ###  https://arxiv.org/pdf/1611.07308
dataset = 'cora' # dataset of paper references
model = 'GAE'

input_dim = 4
hidden1_dim = 32
hidden2_dim = 32
use_feature = True

num_epoch = 800
learning_rate = 0.01 # same as in the paper

# GAE
# Epoch: 0800 train_loss= 0.42650 train_acc= 0.82156 val_roc= 0.97440 val_ap= 0.96066 val_acc= 0.88222 time= 0.18996
# labels_all [1. 1. 1. ... 0. 0. 0.]
# preds_all_zero_one [0 1 1 ... 0 0 0]
# preds_all [0.61183444 0.72740064 0.70234786 ... 0.60457027 0.57231474 0.61847979]
# End of training! test_roc= 0.97432 test_ap= 0.96350 test_acc= 0.88450

# VGAE
# Epoch: 0800 train_loss= 0.54076 train_acc= 0.76804 val_roc= 0.88721 val_ap= 0.83824 val_acc= 0.82038 time= 0.20053
# labels_all [1. 1. 1. ... 0. 0. 0.]
# preds_all_zero_one [1 1 1 ... 0 0 1]
# preds_all [0.67130348 0.71457384 0.65074666 ... 0.62229824 0.64582837 0.68496239]
# End of training! test_roc= 0.89314 test_ap= 0.84668 test_acc= 0.82384
