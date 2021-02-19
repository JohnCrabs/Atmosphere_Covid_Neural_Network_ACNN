# %tensorflow_version 1.x
import tensorflow as tf
import keras
import cv2
import numpy as np
import os

# ---------------------------------- #

# px_width = px_height = px_size
# px_size X px_size X bands X paradigms

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                            padding="same")(input_tensor)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    # second layer
    x = keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                            padding="same")(x)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x


class MyUNet:
    def __init__(self):
        self.UnetCustMod = None

    def set_uNet(self, width, height, channels, n_filters=16):

        # Build U-Net model
        inputs = keras.layers.Input((height, width, channels))
        # s = keras.layers.Lambda(lambda x: x / 255)(inputs)  # normalize the input
        # s = inputs  # uncomment this if image is not 8-bit integer
        conv1 = conv2d_block(inputs, n_filters=n_filters * 1, kernel_size=3, batchnorm=True)
        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = conv2d_block(pool1, n_filters=n_filters * 2, kernel_size=3, batchnorm=True)
        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = conv2d_block(pool2, n_filters=n_filters * 4, kernel_size=3, batchnorm=True)
        # drop3 = keras.layers.Dropout(0.5)(conv3)
        # pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)
        # conv4 = conv2d_block(pool3, n_filters=n_filters * 8, kernel_size=3, batchnorm=True)
        #
        # # now we start the decoder (i.e. expansive path)
        # u5 = keras.layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(conv4)
        # u5 = keras.layers.merge.concatenate([u5, conv3])
        # u5 = keras.layers.Dropout(0.5)(u5)
        # conv5 = conv2d_block(u5, n_filters=n_filters * 4, kernel_size=3, batchnorm=True)
        #
        u6 = keras.layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(conv3)
        u6 = keras.layers.merge.concatenate([u6, conv2])
        u6 = keras.layers.Dropout(0.5)(u6)
        conv6 = conv2d_block(u6, n_filters=n_filters * 4, kernel_size=3, batchnorm=True)

        u7 = keras.layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(conv6)
        u7 = keras.layers.merge.concatenate([u7, conv1])
        u7 = keras.layers.Dropout(0.5)(u7)
        conv7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=True)

        # outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv7)
        outputs = keras.layers.Conv2D(channels, (1, 1), padding='same', activation='sigmoid')(
            conv7)  # change this according to the num of classes
        self.UnetCustMod = keras.Model(inputs=[inputs], outputs=[outputs])

        # the following was the first approach, with custom f1 score as metrics
        # UnetCustMod.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[f1_score])
        # this is the approach of Yu Li
        self.UnetCustMod.compile(optimizer='sgd', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

        self.UnetCustMod.summary()

    def train_uNet(self, X_train, Y_train, X_val, Y_val, export_model_path, export_model_name_path):
        callbacksOptions = [
            keras.callbacks.EarlyStopping(patience=15, verbose=1),
            keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=12, min_lr=0.0001, verbose=1),
            keras.callbacks.ModelCheckpoint(export_model_path, verbose=1,
                                            save_best_only=True,
                                            save_weights_only=True)
        ]

        print('X_train = ', X_train.shape)
        print('X_train = ', Y_train.shape)
        print('X_val = ', X_val.shape)
        print('X_val = ', Y_val.shape)

        results = self.UnetCustMod.fit(X_train, Y_train, batch_size=4, epochs=100, callbacks=callbacksOptions, validation_data=(X_val, Y_val))
        self.UnetCustMod.save(export_model_name_path)

        return results
