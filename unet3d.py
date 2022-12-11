import keras
from keras import layers, utils
import numpy as np
import xarray as xr
import keras.backend as K


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv3D(32, 3, strides=2, padding="same")(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128]:
        x = layers.Activation("relu")(x)
        x = layers.Conv3D(filters, 3, padding="same")(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv3D(filters, 3, padding="same")(x)
        x = layers.Dropout(0.5)(x)

        x = layers.MaxPooling3D(2, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv3D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv3DTranspose(filters, 3, padding="same")(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv3DTranspose(filters, 3, padding="same")(x)
        x = layers.Dropout(0.5)(x)

        x = layers.UpSampling3D(2)(x)

        # Project residual
        residual = layers.UpSampling3D(2)(previous_block_activation)
        residual = layers.Conv3D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv3D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1-((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


if __name__ == '__main__':
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    # model = get_model(img_size, num_classes)
    num_classes = 4
    label = np.load('data/eddy_label_2016.npy')
    vor = np.load('data/vor_gray_label_2016.npy')
    vor = np.expand_dims(np.swapaxes(vor, 1, 3), -1) / 255
    label = np.expand_dims(np.swapaxes(label, 1, 3), -1)  # （N, 14, 260, 250) -> (N, 250, 260, 14, 1)
    label = np.where(vor==1, 3, label)
    label = utils.to_categorical(label, num_classes=num_classes)
    
    x, y, d = label.shape[1:4]
    ds = xr.Dataset(
        {
            "trainX": (["sample", "x", "y", "depth", "channel"], vor),
            "trainY": (["sample", "x", "y", "depth", "classes"], label),
        }
    )
    new_xy = np.linspace(0, x-1, 256)
    new_depth = np.linspace(0, d-1, 16)
    dsi = ds.interp(x=new_xy, y=new_xy, depth=new_depth,method='nearest')

    model = get_model((256, 256, 16, 1), num_classes)
    model.summary()
    callbacks = [
        keras.callbacks.ModelCheckpoint("3d_vor_eddy.h5", save_best_only=True),
    ]
    model.compile(optimizer="adadelta", loss=dice_coefficient)
    model.fit(dsi['trainX'].values, dsi['trainY'].values, epochs=100, batch_size=1, validation_split=0.1, callbacks=callbacks)