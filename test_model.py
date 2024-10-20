import numpy as np
from get_model import get_model
from predict import predict

def test_model():
    # Load the model
    model = get_model(input_shape=(150, 150, 3), num_classes=120)
    model.load_weights('Data/Model/game_bot_model.weights.h5')

    # Create a random input for testing
    test_input = np.random.rand(1, 150, 150, 3)

    # Make a prediction
    prediction = predict(model, test_input)

    print(f'Prediction shape: {prediction.shape}')
    print(f'Prediction sum: {np.sum(prediction)}')
    print(f'Prediction sample: {prediction[0][:5]}')

if __name__ == "__main__":
    test_model()
