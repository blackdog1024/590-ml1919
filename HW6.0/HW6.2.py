from keras.models import Model
from keras import layers
import matplotlib.pyplot as plt
#GET DATASET
from keras.datasets import mnist
import numpy as np
import keras
#GET DATASET
from keras.datasets import mnist
(X, Y), (test_images, test_labels) = mnist.load_data()
from keras.datasets import fashion_mnist
(X_f, Y_f), (test_images_f, test_labels_f) = fashion_mnist.load_data()
#NORMALIZE AND RESHAPE
X=X/np.max(X)
X_f=X_f/np.max(X_f)
test_images = test_images/np.max(test_images)
#The encoding process
input_img = layers.Input(shape=(28,28,1))

#conv1#
x = layers.Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', padding = 'same')(input_img)
x = layers.MaxPooling2D(pool_size = (2,2), padding = 'same')(x)

#conv2#
x = layers.Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
x = layers.MaxPooling2D(pool_size = (2,2), padding = 'same')(x)

#conv3#
x = layers.Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
encoded = layers.MaxPooling2D(pool_size = (2,2), padding = 'same')(x)

#deconv1
x = layers.Conv2D(8, (3,3), activation= 'relu',padding = 'same')(encoded)
x = layers.UpSampling2D((2,2))(x)

#deconv2
x = layers.Conv2D(8, (3,3), activation= 'relu',padding = 'same')(x)
x = layers.UpSampling2D((2,2))(x)

#deconv3
x = layers.Conv2D(16, (3,3), activation= 'relu')(x)
x = layers.UpSampling2D((2,2))(x)
decoded = layers.Conv2D(1, (3,3), activation = 'linear', padding = 'same')(x)

model = keras.Model(input_img, decoded)
model.compile(optimizer='rmsprop',
                loss='mean_squared_error',
              metrics=['accuracy'])

history = model.fit(X, X, epochs=10, batch_size=128,validation_split=0.2)
model.save('convolutional_AE.h5')
threshold,acc = model.evaluate(X,X,batch_size=X.shape[0])
threshold = 4 * threshold
#compute test accuracy
test_loss, test_acc = model.evaluate(test_images, test_images,batch_size=test_images.shape[0])
print("----------test accurancy------------")
print(test_acc)
#EXTRACT MIDDLE LAYER (REDUCED REPRESENTATION)
from keras import Model

X1=model.predict(X)


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
    f.savefig('HW6.2-loss history.png', bbox_inches='tight')
    plt.show()

#compute predict and difference between predict and origin
X1=model.predict(X)
X1 = X1.reshape(60000,28,28*1)
err = np.abs(X1 - X);
X1_f = model.predict(X_f)
X1_f = X1_f.reshape(60000,28,28*1)
errf = np.abs(X1_f-X_f)

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
ax[0].imshow(X[11].reshape(28, 28))
ax[1].imshow(X1[11].reshape(28, 28))
ax[2].imshow(X[23].reshape(28, 28))
ax[3].imshow(X1[23].reshape(28, 28))
f.savefig('HW6.2-encoder result.png', bbox_inches='tight')
plt.show()
plot_loss()
