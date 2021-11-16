from keras.models import Model
from keras import layers
import matplotlib.pyplot as plt
#GET DATASET
from keras.datasets import mnist
import numpy as np
import keras
#GET DATASET
from keras.datasets import cifar10
from keras.datasets import cifar100

(X, Y), (test_images, test_labels) = cifar10.load_data()
index = np.where(Y == 9)

from keras.datasets import fashion_mnist
(X_100, Y_100), (test_images_100, test_labels_100) = cifar100.load_data()
#NORMALIZE AND RESHAPE
X=X/np.max(X)

X_100=X_100/np.max(X_100)
test_images = test_images/np.max(test_images)
#The encoding process
input_img = layers.Input(shape=(32,32,3))

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# At this point the representation is (7, 7, 32)

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)


model = keras.Model(input_img, decoded)
model.compile(optimizer='rmsprop',
                loss='mean_squared_error')

history = model.fit(X, X, epochs=10, batch_size=128,validation_split=0.2)
model.save('convolutional_AE_cifar100.h5')
threshold = 5*model.evaluate(X,X,batch_size=X.shape[0])
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
    f.savefig('HW6.3-loss history.png', bbox_inches='tight')
    plt.show()

#compute predict and difference between predict and origin
X1=model.predict(X)
print(X1.shape)
err = np.abs(X1 - X);
X1_100 = model.predict(X_100)
errf = np.abs(X1_100-X_100)

#anomaly detection
sum_cifar10 = 0
sum_cifar100 = 0
count_cifar10 = 0
count_cifar100 = 0
#print(err)
for row in err:
    error = np.mean(row)
    sum_cifar10 = sum_cifar10 + 1
    if(error > threshold): count_cifar10 = count_cifar10 + 1
for row in errf:
    error_f = np.mean(row)
    sum_cifar100 = sum_cifar100 + 1
    if(error_f > threshold):count_cifar100 = count_cifar100 + 1
print("anomaly ratio of cifar10: ",count_cifar10/sum_cifar10)
print("anomaly ratio of cifar100: ",count_cifar100/sum_cifar100)
#EXTRACT MIDDLE LAYER (REDUCED REPRESENTATION)
from keras import Model

X1=model.predict(X)

#COMPARE ORIGINAL
f, ax = plt.subplots(4,1)
ax[0].imshow(X[11])
ax[1].imshow(X1[11])
ax[2].imshow(X[23])
ax[3].imshow(X1[23])
f.savefig('HW6.3-encoder result.png', bbox_inches='tight')
plt.show()
plot_loss()
