# Arda Mavi
import os
from keras.models import Model
from keras.optimizers import Adadelta
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout

def save_model(model):
    """
    Save the model architecture and weights to files.

    Args:
    model: Keras model to be saved

    Returns:
    None
    """
    model_dir = os.environ.get('GAME_BOT_MODEL_DIR', 'Data/Model/')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save model architecture
    model_json = model.to_json()
    with open(os.path.join(model_dir, "model.json"), "w") as model_file:
        model_file.write(model_json)

    # Save model weights
    model.save_weights(os.path.join(model_dir, "weights.h5"))
    print('Model and weights saved')
    return

def get_model(input_shape=(150, 150, 3), num_classes=120):
    """
    Create and return the CNN model architecture.

    Args:
    input_shape: Tuple, shape of the input images (default: (150, 150, 3))
    num_classes: Integer, number of output classes (default: 120)

    Returns:
    model: Compiled Keras model
    """
    inputs = Input(shape=input_shape)

    # Convolutional layers
    conv_1 = Conv2D(16, (3,3), strides=(1,1))(inputs)
    act_1 = Activation('relu')(conv_1)
    pooling_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act_1)

    conv_2 = Conv2D(32, (3,3), strides=(1,1))(pooling_1)
    act_2 = Activation('relu')(conv_2)
    pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act_2)

    # Flatten and fully connected layers
    flat_1 = Flatten()(pooling_2)

    fc = Dense(256)(flat_1)
    fc = Activation('relu')(fc)
    fc = Dropout(0.5)(fc)
    fc = Dense(num_classes)(fc)

    outputs = Activation('softmax')(fc)

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    save_model(get_model())
