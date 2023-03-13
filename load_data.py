import tensorflow as tf

# Define the directory where the data is stored
data_dir = "/home/tamdau/Desktop/training/train1/dataset"

# Define the batch size
batch_size = 32

# Define the image dimensions
img_height = 224
img_width = 224

# Define the number of classes
num_classes = 2

# Define the data augmentation strategy
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# Define the training data generator
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
  preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
  validation_split=0.2,
  horizontal_flip=True,
  rotation_range=20,
)

# Load the training data
train_data = train_generator.flow_from_directory(
  data_dir,
  target_size=(img_height, img_width),
  batch_size=batch_size,
  class_mode='binary',
  subset='training')

# Load the validation data
val_data = train_generator.flow_from_directory(
  data_dir,
  target_size=(img_height, img_width),
  batch_size=batch_size,
  class_mode='binary',
  subset='validation')
