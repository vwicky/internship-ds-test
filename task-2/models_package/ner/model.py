from spacy import load as load_model
import spacy
from spacy.util import minibatch
from spacy.training.example import Example

import spacy_lookups_data  # Ensure this package is loaded
from spacy.lookups import Lookups

import random

from ..model_manager import CustomModelInterface, NER_MODEL_PATH
from ...data.ner_train_data import train_data

class NERCustomModel(CustomModelInterface):
  def __init__(self, model_path = NER_MODEL_PATH):
    self.nlp = load_model(model_path)
  
  def fit(self, train_data=train_data):
    self.nlp = spacy.blank('en')
    
    lookups = Lookups()
    lookups.add_table("lexeme_norm", {})  # Empty table if needed
    self.nlp.vocab.lookups = lookups
    
    if 'ner' not in self.nlp.pipe_names:
      self.ner = self.nlp.add_pipe('ner', last=True)
    else:
      self.ner = self.nlp.get_pipe('ner')
      
    for _, annotations in train_data:
      for ent in annotations['entities']:
        if ent[2] not in self.ner.labels:
          self.ner.add_label(ent[2])
          
    other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner']
    with self.nlp.disable_pipes(*other_pipes):
      optimizer = self.nlp.begin_training()
      epochs = 100
      for epoch in range(epochs):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=2)
        for batch in batches:
          examples = []
          for text, annotations in batch:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
          self.nlp.update(examples, drop=0.3, losses=losses)
        print(f"Epoch {epoch + 1}, Losses: {losses}")
    self.nlp.to_disk('../models/custom_ner_model')
    return
    
  def predict(self, text):
    return self.nlp(text).ents