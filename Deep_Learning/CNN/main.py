"""

Convolutional Neural Network (CNN):
-----------------------------------
            Convolutional neural network is a regularized type of feed-forward neural network that learns feature
            engineering by itself via filters optimization. Vanishing gradients and exploding gradients, seen during
            backpropagation in earlier neural networks, are prevented by using regularized weights over fewer
            connections.

            Step 1 : Convolutional Layer
            Step 2 : ReLU (Rectified Linear Unit) Layer
            Step 3 : Pooling Layer (Max pooling, Mean Pooling, Sum Pooling ....... ) Layer
            Step 4 : Flatten Layer
            Step 5 : Fully Connected Layer ( ANN )
            Step 6 : Output Layer (Softmax, Cross Entropy, sigmoid)

    Guide Link:
    ----------
            https://www.superdatascience.com/blogs/the-ultimate-guide-to-convolutional-neural-networks-cnn

"""

# Importing the Libraries
import os
import tensorflow as tf
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

print(tf.__version__)

# Pre Processing

# Preprocessing on Training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    directory='dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Preprocessing  on Test Set
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    directory='dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Building the CNN

# Initialising the CNN
cnn = keras.models.Sequential()

# Step 1 : Convolution Layer (ReLU Layer also will be implemented inside convolutional Layer)
cnn.add(keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu',
    input_shape=[64, 64, 3]
))

# Step 2 : Pooling (Max Pooling)
cnn.add(keras.layers.MaxPooling2D(
    pool_size=2,
    strides=2
))

# Adding second convolutional Layer and Pooling Layer
# No need of adding input_shape in convolution Layer, and it needs to be done only if we using it as first layer
cnn.add(keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu'
))

cnn.add(keras.layers.MaxPooling2D(
    pool_size=2,
    strides=2
))

# Step 3 : Flattening
cnn.add(keras.layers.Flatten())

# Step 4 : Full Connection
cnn.add(keras.layers.Dense(
    units=128,
    activation='relu'
))

# Step 5 : Output Layer
cnn.add(keras.layers.Dense(
    units=1,
    activation='sigmoid'
))

# Training the CNN

# Compile the CNN
cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training the CNN and evaluating it on the test set
cnn.fit(
    x=train_data,
    validation_data=test_data,
    epochs=25
)

# Making the Single prediction on trained cnn model
test_image = image.load_img(
    path="dataset/single_prediction/dog_or_cat_1.jpg",
    target_size=(64, 64)
)

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, 0)

result = cnn.predict(test_image)
if result[0][0] == 1:
    prediction = 'Dog'
else:
    prediction = 'Cat'

print(f"\n--------\nPrediction\n--------\n{prediction}")
