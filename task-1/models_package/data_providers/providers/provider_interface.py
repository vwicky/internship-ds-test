import numpy as np
from abc import ABC, abstractmethod


def clip_array(array) -> np.ndarray:
  array = array.astype(int)
  return np.clip(array, 0, 9).astype(int)

class DataProviderInterface(ABC):
  @classmethod
  @abstractmethod
  def provide_train() -> tuple:
    pass
  
  @classmethod
  @abstractmethod
  def provide_test() -> tuple:
    pass