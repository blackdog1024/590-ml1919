import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from keras import models
from keras import layers

#GET DATASET
from keras.datasets import mnist
(X, Y), (test_images, test_labels) = mnist.load_data()

from keras.datasets import fashion_mnist
(X_f, Y_f), (test_images_f, test_labels_f) = fashion_mnist.load_data()
#NORMALIZE AND RESHAPE
X=X/np.max(X)
X=X.reshape(60000,28*28);
X_f=X_f/np.max(X_f)
X_f=X_f.reshape(60000,28*28);
test_images = test_images/np.max(test_images)
test_images = test_images.reshape(10000,28*28)

#MODEL
n_bottleneck=32

#DEEPER
model = models.Sequential()
NH=200
model.add(layers.Dense(NH, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(n_bottleneck, activation='relu'))
model.add(layers.Dense(28*28,  activation='linear'))



#COMPILE AND FIT
model.compile(optimizer='rmsprop',
                loss='mean_squared_error',
              metrics=['accuracy'])
model.summary()
history = model.fit(X, X, epochs=10, batch_size=1000,validation_split=0.2)
model.save('deep_feed_AE.h5')
threshold,acc = model.evaluate(X,X,batch_size=X.shape[0])
threshold = 4 * threshold
#compute test accuracy
test_loss, test_acc = model.evaluate(test_images, test_images,batch_size=test_images.shape[0])
print("----------test accurancy------------")
print(test_acc)

#EXTRACT MIDDLE LAYER (REDUCED REPRESENTATION)
from keras import Model
extract = Model(model.inputs, model.layers[-2].output) # Dense(128,...)
X1 = extract.predict(X)


# #2D PLOT
# plt.scatter(X1[:,0], X1[:,1], c=Y, cmap='tab10')
# plt.show()
#
# #3D PLOT
# ax = plt.figure(figsize=(16,10)).gca(projection='3d')
# ax.scatter(
#     xs=X1[:,0],
#     ys=X1[:,1],
#     zs=X1[:,2],
#     c=Y,
#     cmap='tab10'
# )
# plt.show()

#plot loss
def plot_loss():
    f = plt.figure()
    f, ax = plt.subplots()
    history_dict = history.history
    loss = history_dict['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    f.savefig('HW6.1-loss history.png', bbox_inches='tight')
    plt.show()


#compute predict and difference between predict and origin
X1=model.predict(X)
err = np.abs(X1 - X);
X1_f = model.predict(X_f)
errf = np.abs(X1_f-X_f)

#RESHAPE
X=X.reshape(60000,28,28); #print(X[0])
X1=X1.reshape(60000,28,28); #print(X[0])
X_f = X_f.reshape(60000,28,28)

#anomaly detection
sum_mnist = 0
sum_fashion = 0
count_mnist = 0
count_fashion = 0
#print(err)
for row in err:
    error = np.mean(row)
    sum_mnist = sum_mnist + 1
    if(error > threshold): count_mnist = count_mnist + 1
for row in errf:
    error_f = np.mean(row)
    sum_fashion = sum_fashion + 1
    if(error_f > threshold):count_fashion = count_fashion + 1
print("anomaly ratio of mnist: ",count_mnist/sum_mnist)
print("anomaly ratio of mnist_fashion: ",count_fashion/sum_fashion)


#COMPARE ORIGINAL
f, ax = plt.subplots(4,1)
I1=11; I2=46
ax[0].imshow(X[I1])
ax[1].imshow(X1[I1])
ax[2].imshow(X[I2])
ax[3].imshow(X1[I2])
f.savefig('HW6.1-encoder result.png', bbox_inches='tight')
plt.show()
plot_loss()