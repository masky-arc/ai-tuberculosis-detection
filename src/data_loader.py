import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_datasets(data_dir: str, img_size=(224, 224), batch_size=32):
    """
    Loads training and validation datasets with augmentation.
    """

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training"
    )

    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation"
    )

    return train_data, val_data
