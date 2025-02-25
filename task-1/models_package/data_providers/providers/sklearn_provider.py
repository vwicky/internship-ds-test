from .provider_interface import DataProviderInterface, clip_array
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

class SklearnMnistProvider(DataProviderInterface):
  def __init__(self, test_size=0.3, random_state=42):
    self.mnist = fetch_openml('mnist_784')
    self.X = self.mnist.data
    self.y = self.mnist.target
  
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
      self.X, self.y, 
      test_size=test_size, 
      random_state=random_state
    ) 
    self.y_train = clip_array(self.y_train)
    self.y_test = clip_array(self.y_test)
    
  def provide_train(self) -> tuple:
    return self.X_train.to_numpy(), self.y_train
  
  def provide_test(self) -> tuple:
    return self.X_test.to_numpy(), self.y_test
  
  # def provide_data_frame(self) -> pd.DataFrame:
  #   ds = pd.DataFrame(self.X)
  #   ds['y'] = self.y
  #   return ds