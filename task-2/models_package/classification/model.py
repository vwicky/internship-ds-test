import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
import kagglehub

from .translate import translate
from ..model_manager import CustomModelInterface, CLASSIFICATION_MODEL_PATH

# python -m models_package.classification.model

class ImageHandler:
  img_height, img_width = 252, 320 # got from the notebook
  
  @staticmethod
  def prepare(img_path):
    new_img_path = pathlib.Path(img_path)

    img = tf.keras.utils.load_img(
        new_img_path, 
        target_size=(ImageHandler.img_height, ImageHandler.img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    return img_array
  
class ClassificationCustomModel(CustomModelInterface):
  class_names = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
  translation = translate
  dataset_path = kagglehub.dataset_download("alessiocorrado99/animals10")
  
  def __init__(self, model_path = CLASSIFICATION_MODEL_PATH):
    self.model = tf.keras.models.load_model(model_path)
  
  def fit(self, batch_size=96, epochs=10):
    data_dir = pathlib.Path(self.dataset_path + '\\raw-img')
    
    img_height = ImageHandler.img_height
    img_width = ImageHandler.img_width
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.35,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.35,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size
    )
    class_names = train_ds.class_names
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    num_classes = len(class_names)
    
    data_augmentation = keras.Sequential([
      layers.RandomFlip(
        "horizontal",
        input_shape=(img_height, img_width, 3)),
      layers.RandomRotation(0.1),
      layers.RandomZoom(0.1),
    ])
    
    self.model = tf.keras.models.Sequential([
      data_augmentation,
      layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(48, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(num_classes)
    ])
    self.model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
    )
    self.model.summary()
    history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    self.model.save('../models/classification_model.keras')
    print('> training finished')                    
    return
    
  def predict(self, img_path, num_of_examples=3) -> list:
    img_array = ImageHandler.prepare(img_path)
    
    predictions = self.model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    most_prob = ClassificationCustomModel.most_probable(score, num_of_examples)
    return most_prob
  
  @staticmethod
  def most_probable(score, num_of_units = 2) -> list:
    score = score.numpy()
    ranks = [(ClassificationCustomModel.translation.get(n), int(s * 100)) for n, s in zip(ClassificationCustomModel.class_names, score)]
    ranks.sort(key=lambda x: x[1], reverse=True)
    return ranks[:num_of_units]
  
