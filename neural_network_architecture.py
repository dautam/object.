import tensorflow as tf


# Define the model architecture
def create_model():
    # Define the input layer
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # Define the convolutional layers
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    # Flatten the output from the convolutional layers
    x = tf.keras.layers.Flatten()(x)

    # Define the dense layers
    x = tf.keras.layers.Dense(units=128, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)

    # Define the output layer
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(x)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
