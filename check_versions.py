import tensorflow as tf

print(f'TensorFlow version: {tf.__version__}')
print(f'Keras version: {tf.keras.__version__}')

# Check if ImageDataGenerator is available
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print("ImageDataGenerator is available in tensorflow.keras.preprocessing.image")
except ImportError:
    print("ImageDataGenerator is not available in tensorflow.keras.preprocessing.image")

# Check alternative import
try:
    from keras.preprocessing.image import ImageDataGenerator
    print("ImageDataGenerator is available in keras.preprocessing.image")
except ImportError:
    print("ImageDataGenerator is not available in keras.preprocessing.image")
