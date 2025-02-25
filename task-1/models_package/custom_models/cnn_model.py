from .classifier_interface import MnistClassifierInterface
import tensorflow as tf
import numpy as np
from ..data_providers.providers.provider_interface import clip_array
from sklearn import metrics

class ConvolutionalNNMnist(MnistClassifierInterface):
  def __init__(self):
    self.model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=[28, 28, 1]),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      
      tf.keras.layers.Conv2D(48, kernel_size=(3, 3)),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(units=1),
    ])
    self.model.summary()
    self.model.compile(
      loss="mae",
      optimizer="adam",
    )
  
  def train(self, X_train, y_train) -> None:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    self.model.fit(X_train, y_train, batch_size=256, epochs=3)
    print("> training ended")
  
  def predict(self, X_test) -> np.ndarray:
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    y_pred = self.model.predict(X_test)
    y_pred = clip_array(y_pred)
    return y_pred
  
  def evaluate(self, X_test, y_test) -> None:
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    y_pred = self.predict(X_test)
    print(metrics.classification_report(y_test, y_pred, zero_division=np.nan))