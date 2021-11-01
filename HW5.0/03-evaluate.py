from keras.models import load_model
import numpy as np
#load model
cnn_model = load_model('1D-CNN.h5')
rnn_model = load_model('RNN.h5')

#load data
novel_data = np.load('novel_data.npz')
novel = novel_data['novel']
label = novel_data['label']
rand_indices = np.random.permutation(np.arange(len(novel)))
CUT=int(0.8 * len(novel));
train_idx, test_idx = rand_indices[:CUT], rand_indices[CUT:]
x_test=novel[test_idx]
x_train=novel[train_idx]
y_test=np.array(label)[test_idx]
y_train=np.array(label)[train_idx]
rand_indices = np.random.permutation(x_train.shape[0])
CUT=int(0.8*x_train.shape[0]);
train_idx, val_idx = rand_indices[:CUT], rand_indices[CUT:]
x_val=x_train[val_idx]; y_val=y_train[val_idx]
x_train=x_train[train_idx]; y_train=y_train[train_idx]
print('input_train shape:', x_train.shape)

#evaluate
cnn_model.evaluate(x_train,y_train)
cnn_model.evaluate(x_val,y_val)
cnn_model.evaluate(x_test,y_test)

rnn_model.evaluate(x_train,y_train)
rnn_model.evaluate(x_val,y_val)
rnn_model.evaluate(x_test,y_test)