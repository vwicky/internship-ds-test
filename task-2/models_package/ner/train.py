import sys
from .model import NERCustomModel

# python -m models_package.ner.train false

def main() -> None:
  args = sys.argv[1:]
  print(args)
  
  if args is None or len(args) < 1:
    pass

  model = NERCustomModel(model_path=None)
  model.fit()
  print("model trained & saved")
  
if __name__ == "__main__":
  main()
  