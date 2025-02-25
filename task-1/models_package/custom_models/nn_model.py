from .classifier_interface import MnistClassifierInterface
import tensorflow as tf
from sklearn import metrics
from ..data_providers.providers.provider_interface import clip_array
import numpy as np

class FeedForwardNNMnist(MnistClassifierInterface):
  def __init__(self):
    self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu',
                              input_shape=[784]),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    self.model.summary()
    self.model.compile(optimizer='adam', loss='mae')  
  
  def train(self, X_train, y_train) -> None:
    self.model.fit(
      X_train, y_train,
      # validation_data=(X_test, y_test),
      batch_size=256, 
      epochs=20,  
    )
    print("> training ended")
  
  def predict(self, X_test) -> np.ndarray:
    y_pred = self.model.predict(X_test)
    y_pred = clip_array(y_pred)
    return y_pred
  
  def evaluate(self, X_test, y_test) -> None:
    y_pred = self.predict(X_test)
    print(metrics.classification_report(y_test, y_pred, zero_division=np.nan))