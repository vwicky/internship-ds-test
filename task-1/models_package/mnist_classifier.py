from .data_providers.data_provider_manager import DataProviderManager
from .custom_models.rf_model import RandomForestMnist
from .custom_models.cnn_model import ConvolutionalNNMnist
from .custom_models.nn_model import FeedForwardNNMnist

class MnistClassifier:
  model_map = {
    'cnn': ConvolutionalNNMnist,
    'nn': FeedForwardNNMnist,
    'rf': RandomForestMnist,
  }
  data_map = {
    'tensorflow': DataProviderManager.request_tensorflow,
    'sklearn': DataProviderManager.request_sklearn,
  }
  def __init__(self, algorithm: str):
    self.model = self.model_map.get(algorithm)()
    self.data_provider = None
    
  def train(self, provider: str) -> None:
    self.data_provider = self.data_map.get(provider)()
    X_train, y_train = self.data_provider.provide_train()
    self.model.train(X_train, y_train)
    
  def predict(self, img_path: str) -> any:
    my_img = DataProviderManager.request_img(src=img_path)
    y_pred = self.model.predict(my_img)
    return y_pred
    
  def evaluate(self) -> None:
    if self.data_provider is None:
      print('train the model first')
      return
    X_test, y_test = self.data_provider.provide_test()
    self.model.evaluate(X_test, y_test)