from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
  @classmethod
  @abstractmethod
  def train() -> None:
    pass
  
  @classmethod
  @abstractmethod
  def predict() -> None:
    pass