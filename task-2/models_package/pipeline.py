from .classification.model import ClassificationCustomModel
from .ner.model import NERCustomModel

class ResultWrapper:
  def __init__(self, res_list):
    self.res = res_list
    
  def __str__(self):
    return '\n'.join(f"> {res[0][0]}, {res[0][1]}% - {'✔' if res[1] else '❌'}" for res in self.res)
  
class CustomPipeline:
  def __init__(self, ner_model = None, classification_model = None):
    self.ner_model = ner_model or NERCustomModel()
    self.classification_model = classification_model or ClassificationCustomModel()
    
  def predict(self, img_path, text, num_probable_guesses=1) -> bool:
    ner_result = self.ner_model.predict(text)
    print(ner_result)
    
    if len(ner_result) <= 0:
      raise LookupError("Couldn't detect animal names")
    
    ner_result = ner_result[0].__str__()
    classification_result = self.classification_model.predict(img_path, num_of_examples=num_probable_guesses)
    
    result = [(res, res[0] == ner_result) for res in classification_result]
    result_bool = result[0][0][0] == ner_result
    print(f"ner_result - {ner_result}, res {result[0][0]}")
    print(ResultWrapper(result))
    return result_bool
  