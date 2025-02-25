import sys
from .model import NERCustomModel

# python -m models_package.ner.inference

def main() -> None:
  args = sys.argv[1:]
  if args is None or len(args) < 1:
    print('bad arguments')
    return
  
  text_input = args[0]
  print(text_input)
  
  model = NERCustomModel()
  answer = model.predict(text=text_input)
  print(f"the text: {text_input} contains {answer}")

if __name__ == '__main__':
  main()