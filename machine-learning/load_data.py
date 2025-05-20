import tensorflow as tf
import keras

def load_data():
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1)
    ])
    
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        "data/images/train",
        image_size=(48, 48),
        color_mode="grayscale",
        batch_size=32,
        label_mode="int"
    )

    validation_dataset = keras.preprocessing.image_dataset_from_directory(
        "data/images/validation",
        image_size=(48, 48),
        color_mode="grayscale",
        batch_size=32,
        label_mode="int"
    )
    
    train_dataset = train_dataset.map(
        lambda x,y: (data_augmentation(x/255.0, training = True ), y)
    )
    
    validation_dataset = validation_dataset.map(
        lambda x, y: (x / 255.0, y)
    )
    
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset, validation_dataset
    


load_data()