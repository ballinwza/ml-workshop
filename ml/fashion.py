import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

# Prepare dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Check shape of data
# print(train_images.shape)
# print(train_labels.shape)
# print(f"Train label \n{train_labels}")

## Normalize data to between 0 and 1
# train_images = train_images / 255.0
# test_images = test_images / 255.0

# print(f"Train image after divide : \n{train_images[0]}")

## Checking image size by matplotlib
def check_image_size(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()

check_image_size(train_images[1])

## Show exmple of images
def show_images(images,labels):
    plt.figure(figsize=(10,10))
    for i in range(len(images)):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()

show_images(train_images[:5], train_labels[:5])

## Try with Pipeline
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None, None, 1)),
    tf.keras.layers.Resizing(28,28),
    tf.keras.layers.Rescaling(1.0/255.0),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10) # train 10 times

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Accuracy : {test_acc}")


predictions = model.predict(test_images)

model.save("models/fashion_model_1.keras")

## Visualize prediction
np.argmax(predictions[0])
test_labels[0]

def plot_predicted_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color=color)

def plot_predicted_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.barh(range(10), predictions_array, color="#777777")
    plt.xlim(0,1)
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    plt.yticks(range(10), class_names)


def multi_plot_predicted():
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_predicted_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_predicted_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()

multi_plot_predicted()