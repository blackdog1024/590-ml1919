import numpy
from keras import preprocessing
import numpy as np
from keras.models import Sequential 
from keras import layers
import matplotlib.pyplot as plt
from keras import regularizers
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import roc_curve,auc
import scipy
from itertools import cycle
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm, datasets
#---------------------------
#USER PARAM
#---------------------------
max_features = 20000    #DEFINES SIZE OF VOCBULARY TO USE
maxlen       = 250      #CUTOFF REVIEWS maxlen 20 WORDS)
epochs       = 20
batch_size   = 50
verbose      = 1
embed_dim    = 8        #DIMENSION OF EMBEDING SPACE (SIZE OF VECTOR FOR EACH WORD)
lr           = 0.001    #LEARNING RATE
regularizers.l1(0.001)
regularizers.l1_l2(l1=0.0001, l2=0.0001)
#---------------------------
#GET AND SETUP DATA
#---------------------------
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

print(x_train[0][0:10]) # ,y_train.shape)

#truncating='pre' --> KEEPS THE LAST 20 WORDS
#truncating='post' --> KEEPS THE FIRST 20 WORDS
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen,truncating='post')
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen,truncating='post')
# print('input_train shape:', x_train.shape)
print(x_train[0][0:10]) # ,y_train.shape)
# print('input_train shape:', x_train.shape)
x_train = numpy.array(x_train)
x_test = numpy.array(x_test)
#PARTITION DATA

rand_indices = np.random.permutation(x_train.shape[0])
CUT=int(0.8*x_train.shape[0]);
train_idx, val_idx = rand_indices[:CUT], rand_indices[CUT:]

x_val=x_train[val_idx]; y_val=y_train[val_idx]
x_train=x_train[train_idx]; y_train=y_train[train_idx]
print('input_train shape:', x_train.shape)


#---------------------------
#plotting function
#---------------------------
def report(model,history,title='',I_PLOT=True):

    print(title+": TEST METRIC (loss,accuracy):",model.evaluate(x_test,y_test,batch_size=50000,verbose=verbose))

    if(I_PLOT):
        #PLOT HISTORY
        epochs = range(1, len(history.history['loss']) + 1)
        plt.figure()
        plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
        plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')

        plt.plot(epochs, history.history['acc'], 'ro', label='Training acc')
        plt.plot(epochs, history.history['val_acc'], 'r', label='Validation acc')

        plt.title(title)
        plt.legend()
        plt.savefig('HISTORY-' + title + '.png')  # save the figure to file
        plt.show()


def ROC(type):
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(
        svm.SVC(kernel="linear", probability=True)
    )
    y_score = classifier.fit(x_train, y_train).decision_function(x_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = label.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if type == "RNN":plt.title("RNN-Some extension of Receiver operating characteristic to multiclass")
    if type == "CNN": plt.title("RNN-Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.savefig('ROC_AUC.png')  # save the figure to file
    plt.show()



print("---------------------------")
print("LSTM")
print("---------------------------")

rnn_model = Sequential()
rnn_model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
rnn_model.add(layers.LSTM(32))
rnn_model.add(layers.Dropout(0.5))
rnn_model.add(layers.Dense(3, activation='sigmoid'))
rnn_model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc'])
rnn_model.summary()
history = rnn_model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
report(rnn_model,history,title="LTSM")
rnn_model.save('RNN.h5')
ROC("RNN")

# print("---------------------------")
# print("SimpleRNN")
# print("---------------------------")
#
# model = Sequential()
# model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
# model.add(layers.SimpleRNN(32))
# model.add(layers.Dense(3, activation='softmax'))
# model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['acc'])
# model.summary()
# history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
# report(history,title="SimpleRNN")



print("---------------------------")
print("1D-CNN")
print("---------------------------")

cnn_model = Sequential()
cnn_model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
cnn_model.add(layers.Conv1D(128, 10, activation='relu',))
cnn_model.add(layers.MaxPooling1D(5))
cnn_model.add(layers.Conv1D(128, 10, activation='relu'))
cnn_model.add(layers.GlobalMaxPooling1D())
cnn_model.add(layers.Dense(3,activation='softmax',kernel_regularizer=regularizers.l1_l2(0.0001)))
cnn_model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['acc'])

cnn_model.summary()
history = cnn_model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
report(cnn_model,history,title="CNN")
cnn_model.save('1D-CNN.h5')
ROC("CNN")













# print(np.array([1,0,0,0,0,0,0]).reshape(7,1))
# exit()


# test=preprocessing.sequence.pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7, 8]],padding='post', maxlen=3)
# print(test)












# from keras.datasets import imdb
# from keras.preprocessing import sequence
# from keras import layers
# from keras.models import Sequential


# max_features = 10000
# maxlen = 500

# (x_train, y_train), (x_test, y_test) = imdb.load_data(
#     num_words=max_features)


# def report():




  
# #NO BIDIRECTIONS






# exit()

# #REVERSE
# x_train = [x[::-1] for x in x_train]
# x_test = [x[::-1] for x in x_test]


# #NO BIDIRECTIONS
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# model = Sequential()
# model.add(layers.Embedding(max_features, 128))
# model.add(layers.LSTM(32))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['acc'])

# history = model.fit(x_train, y_train,
#                     epochs=10,
#                     batch_size=128,
#                     validation_split=0.2)

# #BI-DIRECTIONSAL 
# model = Sequential() 
# model.add(layers.Embedding(max_features, 32)) 
# model.add(layers.Bidirectional(layers.LSTM(32))) 
# model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 
# history = model.fit(x_train, y_train,
# epochs=10, batch_size=128, validation_split=0.2)


# from keras.models import Sequential
# from keras import layers
# from keras.optimizers import RMSprop
# model = Sequential()
# model.add(layers.Bidirectional(
#     layers.GRU(32), input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=40,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)
