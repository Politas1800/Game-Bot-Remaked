# Arda Mavi
import os
import numpy as np
import tensorflow as tf
from get_dataset import get_dataset
from get_model import get_model, save_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Training hyperparameters
epochs = 50  # Increased for better learning

def create_dataset(X, Y, batch_size, is_training=False):
    print(f"Creating dataset with X shape: {X.shape}, Y shape: {Y.shape}")
    num_samples = len(X)
    batch_size = min(batch_size, num_samples)  # Ensure batch_size doesn't exceed num_samples

    # Check for data shape consistency
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Mismatch in number of samples: X has {X.shape[0]}, Y has {Y.shape[0]}")

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))

    # Data augmentation for training
    if is_training:
        def augment(x, y):
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_brightness(x, max_delta=0.2)
            x = tf.image.random_contrast(x, lower=0.8, upper=1.2)
            return x, y

        dataset = dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Debug: Check multiple samples from the dataset
    print("Checking dataset samples:")
    for i, (x, y) in enumerate(dataset.take(3)):
        print(f"Batch {i}:")
        print(f"  X shape: {x.shape}, Y shape: {y.shape}")
        print(f"  X min: {tf.reduce_min(x)}, max: {tf.reduce_max(x)}")
        print(f"  Y: {y[:5]}")  # Print first 5 labels of each batch
        print(f"  Y unique values: {tf.unique(tf.reshape(y, [-1]))[0]}")

    return dataset

def train_model(model, train_dataset, val_dataset, steps_per_epoch, validation_steps, class_weights):
    """
    Train the model using tf.data.Dataset.

    Args:
    model: The model to be trained
    train_dataset: Dataset for training data
    val_dataset: Dataset for validation data
    steps_per_epoch: Number of steps per epoch
    validation_steps: Number of validation steps
    class_weights: Dictionary of class weights

    Returns:
    model: Trained model
    """
    checkpoints = []
    checkpoint_dir = os.environ.get('GAME_BOT_CHECKPOINT_DIR', 'Data/Checkpoints/')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # ModelCheckpoint to save the best model
    checkpoints.append(ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'best_weights.weights.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='auto'
    ))

    # TensorBoard callback for visualization
    checkpoints.append(TensorBoard(
        log_dir=os.path.join(checkpoint_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        write_images=False
    ))

    # Early stopping to prevent overfitting
    checkpoints.append(EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ))

    # Train the model
    try:
        history = model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=checkpoints,
            class_weight=class_weights
        )
        print("Training completed successfully.")
        print(f"Training history: {history.history}")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

    return model

def main():
    """
    Main function to orchestrate the training process.
    """
    try:
        # Load dataset
        X, X_test, Y, Y_test = get_dataset()

        # Print shape and unique values of X and Y
        print(f"Shape of X: {X.shape}")
        print(f"Shape of Y: {Y.shape}")
        print(f"Unique values in Y: {np.unique(Y)}")
        print(f"Min value in X: {X.min()}, Max value in X: {X.max()}")

        # Print a small sample of the data
        print("Sample of X:")
        print(X[:5])
        print("Sample of Y:")
        print(Y[:5])

        num_classes = Y.shape[1]
        print(f"Number of classes: {num_classes}")

        # Calculate class weights for multi-class classification
        y_integers = np.argmax(Y, axis=1)
        unique_classes = np.unique(y_integers)
        print(f"Unique classes in y_integers: {unique_classes}")
        class_weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_integers)
        class_weights = dict(zip(unique_classes, class_weights))

        # Print calculated class weights
        print(f"Calculated class weights: {class_weights}")

        # Determine batch size based on dataset size
        batch_size = max(1, min(32, len(X) // 4))  # Use at most 32 or quarter of the dataset size, but at least 1
        print(f"Using batch size: {batch_size}")

        # Ensure we have enough data for training and validation
        if len(X) < 2 * batch_size:
            print(f"Warning: Not enough data for training. Reducing batch size to {len(X) // 2}")
            batch_size = max(1, len(X) // 2)

        if len(X_test) < batch_size:
            print(f"Warning: Not enough data for validation. Using all test data for each batch.")
            batch_size_val = len(X_test)
        else:
            batch_size_val = batch_size

        # Create data generators
        train_dataset = create_dataset(X, Y, batch_size, is_training=True)
        val_dataset = create_dataset(X_test, Y_test, batch_size_val)

        # Calculate steps per epoch and validation steps
        steps_per_epoch = max(1, len(X) // batch_size)
        validation_steps = max(1, len(X_test) // batch_size_val)

        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")

        # Create a new model for multi-class classification
        model = get_model(input_shape=(150, 150, 3), num_classes=num_classes)

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Print model summary
        model.summary()

        # Train model
        model = train_model(model, train_dataset, val_dataset, steps_per_epoch, validation_steps, class_weights)

        # Save trained model
        model_save_path = os.path.join('Data', 'Model', 'game_bot_model.weights.h5')
        model.save_weights(model_save_path)
        print(f"Model weights saved to {model_save_path}")

        print("Model training and saving completed successfully.")
        return model
    except Exception as e:
        print(f"An error occurred in the main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()
