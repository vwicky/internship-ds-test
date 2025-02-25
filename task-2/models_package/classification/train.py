import sys
from .model import ClassificationCustomModel

# python -m models_package.classification.train

def main() -> None:
  args = sys.argv[1:]
  print(args)
  
  model = ClassificationCustomModel()
  
  if args is None or len(args) < 1:
    pass
  
  if len(args) == 2:
    batch_size, epochs = args[0], args[1]
    model.fit(batch_size=batch_size, epochs=epochs)
  elif len(args) == 1:
    batch_size = args[0]
    model.fit(batch_size=batch_size)
  else:
    model.fit()
  print("model trained & saved")
  
if __name__ == "__main__":
  main()
  