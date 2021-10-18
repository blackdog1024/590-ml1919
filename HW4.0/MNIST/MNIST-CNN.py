
#MODIFIED FROM CHOLLETT P120
import tensorflow as tf
import keras.datasets.cifar10
from keras import layers 
from keras import models
import numpy as np
import warnings
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras import optimizers
from keras.models import load_model
warnings.filterwarnings("ignore")

#hyper parameters
cifar10_size = 50000.
MNIST_size = 60000.
test_size = 10000
NKEEP=10000
batch_size=int(0.05*NKEEP)
epochs=20

#-------------------------------------
#flag of dataset, which dataset used can be changed here
#-------------------------------------
#FLAG = cifar10
FLAG = mnist
#FLAG = fashion_mnist

#flag of data_augmentation
data_aumentation = False
#data_aumentation = True
#-------------------------------------
#define the model type
#-------------------------------------
model_type = "CNN"
#model_type = "ANN"
def save_model(model, name):
    # save the model
    model.save(name)

def read_model(name):
    model = load_model(name)
    model.summary()

def visualize():
    img = train_images[0]
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    layer_outputs = [layer.output for layer in model.layers[:3]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    # Visualizing every channel in every intermediate activation
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()




if(FLAG == cifar10):
    (train_images,train_labels),(test_images,test_labels) = keras.datasets.cifar10.load_data()
else:
    (train_images, train_labels), (test_images, test_labels) = FLAG.load_data()

#spilt validation data
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2)
print("train_images shape:", train_images.shape)
print("test_images shape:", test_images.shape)

#visulize a image
def visualize_image():
    image=train_images[0]
    plt.imshow(image, cmap=plt.cm.gray);
    plt.show()

visualize_image()

#-------------------------------------
#BUILD MODEL SEQUENTIALLY (LINEAR STACK)
#-------------------------------------

model = models.Sequential()
if(model_type == "CNN"):
    if(FLAG == cifar10):
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    else:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()

if(model_type == "ANN"):
    if (FLAG == cifar10):
        model.add(layers.Dense(512, activation='relu', input_shape=(32 * 32 * 3,)))
        model.add(layers.Dense(10, activation='softmax'))
    else:
        model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
        model.add(layers.Dense(10, activation='softmax'))


#-------------------------------------
#REFORMAT DATA
#-------------------------------------
if(model_type == 'CNN'):
    if(FLAG == cifar10):
        train_images = train_images.reshape((int(cifar10_size * 0.8), 32, 32, 3))
        val_images = val_images.reshape((int(cifar10_size * 0.2), 32, 32, 3))
        test_images = test_images.reshape((test_size, 32, 32, 3))
    else:
        train_images = train_images.reshape((int(MNIST_size * 0.8), 28, 28, 1))
        val_images = val_images.reshape((int(MNIST_size * 0.2), 28, 28, 1))
        test_images = test_images.reshape((test_size, 28, 28, 1))
if(model_type == 'ANN'):
    NKEEP = 60000.
    epochs = 50
    if(FLAG == cifar10):
        train_images = train_images.reshape((int(cifar10_size * 0.8), 32 * 32 * 3))
        #train_images = train_images.astype('float32') / train_images.max()
        val_images = val_images.reshape((int(cifar10_size * 0.2), 32 * 32 * 3))
        #val_images = val_images.astype('float32') / val_images.max()
        test_images = test_images.reshape((test_size, 32 * 32 * 3))
        #test_images = test_images.astype('float32') / test_images.max()
    else:
        train_images = train_images.reshape((int(MNIST_size * 0.8), 28 * 28 * 1))
        train_images = train_images.astype('float32') / train_images.max()
        val_images = val_images.reshape((int(MNIST_size * 0.2), 28 * 28 * 1))
        val_images = val_images.astype('float32') / val_images.max()
        test_images = test_images.reshape((test_size, 28 * 28 * 1))
        test_images = test_images.astype('float32') / test_images.max()



#NORMALIZE
train_images = train_images.astype('float32') / 255
val_images = val_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
#DEBUGGING

print("batch_size",batch_size)
#rand_indices = np.random.permutation(train_images.shape[0])

# exit()


#CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
tmp=train_labels[0]
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)
print(tmp, '-->',train_labels[0])
print("train_labels shape:", train_labels.shape)
print("val_labels shape:", val_labels.shape)
#-------------------------------------
#COMPILE AND TRAIN MODEL
#-------------------------------------
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#-------------------------------------
#data augmentation
#-------------------------------------
if((model_type =='CNN') & (data_aumentation == True)):
    #Training the convnet using data-augmentation generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,)
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow(train_images, train_labels,batch_size = batch_size, shuffle=True, seed = None)
    val_generator = val_datagen.flow(val_images, val_labels,batch_size = batch_size, shuffle=True, seed = None )
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_images)/batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_images)/batch_size)

    #save the model
    save_model(model,'data_augmentation.h5')

else:
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data= (val_images,val_labels))
#
#
#-------------------------------------
#EVALUATE ON TEST DATA
#-------------------------------------
train_loss, train_acc = model.evaluate(train_images, train_labels, batch_size=batch_size)
val_loss,val_acc = model.evaluate(val_images, val_labels, batch_size=batch_size)
test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=test_images.shape[0])
print('train_acc:', train_acc)
print('test_acc:', test_acc)

#-------------------------------------
#visualize loss and accurancy
#-------------------------------------
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b-', label='Training loss')
plt.plot(epochs, val_loss, 'r-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'b-', label='Training acc')
plt.plot(epochs, val_acc, 'r-', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


visualize()

#print (train/test/val) metrics
print('train accurancy')
print(acc)
print('train loss')
print(loss)
print('validaton accurancy')
print(val_acc)
print('validaton loss')
print(val_loss)
print('test accurancy')
print(test_acc)
print('test loss')
print(test_loss)
