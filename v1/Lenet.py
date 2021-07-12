import tensorflow as tf

LeNet_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=5, kernel_size=(5, 5), strides=(1, 1), input_shape=(28, 28, 1), padding='valid',
                           activation='relu', kernel_initializer='uniform'),
    tf.keras.layers.MaxPool2D((2, 2)),

    tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu',
                           kernel_initializer='uniform'),
    tf.keras.layers.MaxPool2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(120, activation='relu'),

    tf.keras.layers.Dense(84, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax'),

])

LeNet_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

