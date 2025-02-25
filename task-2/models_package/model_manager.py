from abc import ABC, abstractmethod

NER_MODEL_PATH = 'models/custom_ner_model'
CLASSIFICATION_MODEL_PATH = 'models/classification_model.keras'

class CustomModelInterface(ABC):
  @classmethod
  @abstractmethod
  def fit():
    pass
  
  @classmethod
  @abstractmethod
  def predict():
    pass