import numpy as np
from PIL import Image

from .providers.sklearn_provider import SklearnMnistProvider
from .providers.tensorflow_provider import TensorflowMnistProvider

class DataProviderManager:
  """
    Small research showed that there are two libs for MNIST -- at least two libs like tf and sklearn. I couldn't really
    choose one specific so I wrote this simple wrapper that gives you a dataset that
    you want. Now, RF implemented with sklearn can be trained on TF dataset. 
  """
  @staticmethod
  def request_sklearn():
    return SklearnMnistProvider()
  
  @staticmethod
  def request_tensorflow():
    return TensorflowMnistProvider()
  
  @staticmethod
  def request_img(src: str) -> np.ndarray:
    """
      turns an image into an ndarray with length = 784. Made it to test my own
      written digits. Keeping it fun or smth
    """
    img = Image.open(fp=src)
    img = img.resize((28, 28), Image.LANCZOS)  # best down-sizing filter
    img = img.convert('L')  # convert the image to *greyscale*
    img = np.array(img)
    img = img.reshape(1, 28 * 28)
    return img