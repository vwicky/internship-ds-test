import sys
from .model import ClassificationCustomModel

# python -m models_package.classification.train {true | Any}

def main() -> None:
  args = sys.argv[1:]
  print(args)
  
  if args is None or len(args) < 1:
    pass
  
  model = None
  if len(args) == 1:
    pretrained = args[0]
    if pretrained.lower() == 'true': 
      model = ClassificationCustomModel()
    else:
      model = ClassificationCustomModel(model_path=None)
  model.fit()
  print("model trained & saved")
  
if __name__ == "__main__":
  main()
  