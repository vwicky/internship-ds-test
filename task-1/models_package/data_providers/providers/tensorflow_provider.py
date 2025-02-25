from .provider_interface import DataProviderInterface
import tensorflow as tf

class TensorflowMnistProvider(DataProviderInterface):
  def __init__(self):
    self.mnist = tf.keras.datasets.mnist
    (X_train, self.y_train), (X_test, self.y_test) = self.mnist.load_data()
    self.X_train = X_train.reshape(X_train.shape[0], -1)
    self.X_test = X_test.reshape(X_test.shape[0], -1)
    
  def provide_train(self) -> tuple:
    return self.X_train, self.y_train
  
  def provide_test(self) -> tuple:
    return self.X_test, self.y_test