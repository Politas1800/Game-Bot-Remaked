# Arda Mavi
import os
import platform
import numpy as np
from time import sleep
# from PIL import ImageGrab  # Commented out as it requires GUI
from game_control import *
from predict import predict
from keras.models import model_from_json

# Mock function to replace ImageGrab
def mock_image_grab():
    """
    Mock function to simulate screen capture in a headless environment.
    Returns a numpy array of shape (height, width, 3) filled with random values.
    """
    return np.random.randint(0, 256, size=(1080, 1920, 3), dtype=np.uint8)

def main():
    """
    Main function to run the AI game bot.

    This function loads the trained model, continuously captures screenshots,
    predicts actions based on the screenshots, and executes those actions.
    """
    # Get model and weights file paths from environment variables
    model_path = os.environ.get('GAME_BOT_MODEL_PATH', 'Data/Model/model.json')
    weights_path = os.environ.get('GAME_BOT_WEIGHTS_PATH', 'Data/Model/weights.h5')

    # Check if environment variables are set
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please set the GAME_BOT_MODEL_PATH environment variable.")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}. Please set the GAME_BOT_WEIGHTS_PATH environment variable.")

    # Load the model
    with open(model_path, 'r') as model_file:
        model = model_file.read()
    model = model_from_json(model)
    model.load_weights(weights_path)

    print('AI start now!')

    while True:
        # Capture screenshot using mock function
        screen = mock_image_grab()

        # Predict action based on the screenshot
        Y = predict(model, screen)

        if Y == [0,0,0,0]:
            # No action predicted
            continue
        elif Y[0] == -1 and Y[1] == -1:
            # Keyboard action only
            key = get_key(Y[3])
            if Y[2] == 1:
                # Press key
                press(key)
            else:
                # Release key
                release(key)
        elif Y[2] == 0 and Y[3] == 0:
            # Mouse action only
            click(Y[0], Y[1])
        else:
            # Both mouse and keyboard action
            # Execute mouse action
            click(Y[0], Y[1])
            # Execute keyboard action
            key = get_key(Y[3])
            if Y[2] == 1:
                # Press key
                press(key)
            else:
                # Release key
                release(key)

        # Add a small delay to prevent excessive CPU usage
        sleep(0.1)

if __name__ == '__main__':
    main()
