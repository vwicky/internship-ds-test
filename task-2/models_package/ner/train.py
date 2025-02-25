import sys
from .model import NERCustomModel

# python -m models_package.ner.train

def main() -> None:
  args = sys.argv[1:]
  print(args)
  
  model = NERCustomModel()
  
  if args is None or len(args) < 1:
    pass
  
  elif len(args) == 1:
    train_data = args[0]
    model.fit(train_data=train_data)
  else:
    model.fit()
  print("model trained & saved")
  
if __name__ == "__main__":
  main()
  