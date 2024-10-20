# Arda Mavi
import os
import sys
import numpy as np
from time import sleep
from game_control import get_id
from get_dataset import save_img

# Mock function to replace ImageGrab
def mock_image_grab():
    """
    Mock function to simulate screen capture in a headless environment.
    Returns a numpy array of shape (150, 150, 3) filled with random values.
    """
    return np.random.randint(0, 256, size=(150, 150, 3), dtype=np.uint8)

def get_screenshot():
    img = mock_image_grab()
    img = img.astype('float32') / 255.
    return img

def save_event_keyboard(data_path, event, key, counter):
    try:
        key = get_id(key)
        file_path = data_path + '/-1,-1,{0},{1}_{2}.png'.format(event, key, counter)
        screenshot = get_screenshot()
        save_img(screenshot, file_path)
        print(f"Successfully saved keyboard event: {file_path}")
        return True
    except Exception as e:
        print(f"Error saving keyboard event: {e}")
        return False

def save_event_mouse(data_path, x, y):
    file_path = data_path + '/{0},{1},0,0.png'.format(x, y)
    screenshot = get_screenshot()
    save_img(screenshot, file_path)
    return

def simulate_mouse_events(num_events=400):
    data_path = 'Data/Train_Data/Mouse'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    for _ in range(num_events):
        x = np.random.randint(0, 1920)
        y = np.random.randint(0, 1080)
        save_event_mouse(data_path, x, y)
        sleep(0.1)  # Simulate delay between events

def simulate_keyboard_events(num_events=400):
    data_path = 'Data/Train_Data/Keyboard'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    keys = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']  # Example keys
    counter = 0
    successful_saves = 0
    for _ in range(num_events):
        event = np.random.choice([1, 2])  # 1 for press, 2 for release
        key = np.random.choice(keys)
        if save_event_keyboard(data_path, event, key, counter):
            successful_saves += 1
            counter += 1
        sleep(0.1)  # Simulate delay between events
    print(f"Total keyboard events successfully saved: {successful_saves}")

def main():
    dataset_path = 'Data/Train_Data/'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    print("Simulating mouse events...")
    simulate_mouse_events()
    print("Simulating keyboard events...")
    simulate_keyboard_events()
    print("Dataset creation complete.")
    return

if __name__ == '__main__':
    main()
