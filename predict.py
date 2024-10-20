# Arda Mavi
import numpy as np

def predict(model, X):
    """
    Predict the output using the trained model.

    Args:
    model: Trained Keras model
    X: Input image as numpy array with shape (1, 150, 150, 3)

    Returns:
    Y: Predicted output
    """
    # Input validation
    if X.shape != (1, 150, 150, 3):
        raise ValueError(f"Expected input shape (1, 150, 150, 3), but got {X.shape}")

    # Normalize the input
    X = X.astype('float32') / 255.

    # Make prediction
    Y = model.predict(X)
    return Y
