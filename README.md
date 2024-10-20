# AI Game Bot

## Overview

This project is an AI game bot that uses computer vision and machine learning techniques to play a game. The bot captures screenshots of the game, processes the images, and predicts actions to be taken in the game. The project includes scripts for dataset creation, model training, and prediction.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Politas1800/Game-Bot-Remaked.git
   cd Game-Bot-Remaked
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the AI Game Bot

1. Set up the environment variables for the model and weights file paths:
   ```bash
   export GAME_BOT_MODEL_PATH=path/to/model.json
   export GAME_BOT_WEIGHTS_PATH=path/to/weights.h5
   ```

2. Run the AI game bot:
   ```bash
   python ai.py
   ```

### Creating the Dataset

1. Run the script to create the dataset:
   ```bash
   python create_dataset.py
   ```

### Training the Model

1. Run the script to train the model:
   ```bash
   python train.py
   ```

### Analyzing the Dataset

1. Run the script to analyze the dataset:
   ```bash
   python analyze_dataset.py
   ```

### Checking Versions

1. Run the script to check the versions of TensorFlow and Keras:
   ```bash
   python check_versions.py
   ```

## Script Descriptions

- `ai.py`: Main script to run the AI game bot. Captures screenshots, predicts actions, and executes those actions.
- `create_dataset.py`: Script to create the dataset by simulating mouse and keyboard events and capturing screenshots.
- `train.py`: Script to train the model using the created dataset.
- `analyze_dataset.py`: Script to analyze the dataset, including shapes and unique values.
- `check_versions.py`: Script to check the versions of TensorFlow and Keras.
- `get_dataset.py`: Script to load and preprocess the dataset.
- `get_model.py`: Script to create and save the model architecture and weights.
- `predict.py`: Script to make predictions using the trained model.
- `game_control.py`: Script containing mock functions for game control without GUI dependencies.

## Environment Variables

- `GAME_BOT_MODEL_PATH`: Path to the model JSON file.
- `GAME_BOT_WEIGHTS_PATH`: Path to the model weights file.
- `GAME_BOT_MODEL_DIR`: Directory to save the model architecture and weights.
- `GAME_BOT_CHECKPOINT_DIR`: Directory to save the model checkpoints.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository.

Please ensure your code follows the project's coding standards and includes appropriate tests.

Thank you for contributing!
