from .model import ClassificationCustomModel
import sys

# python -m models_package.classification.inference data/animals_test_img/dog.jpeg

def main() -> None:
  args = sys.argv[1:]
  print(args)
  if args is None or len(args) < 1:
    print('bad arguments')
    return
  
  model = ClassificationCustomModel()
  img_path_input = args[0]
  
  answer = None
  if len(args) == 2:
    num_og_guesses = args[1]
    answer = model.predict(
      img_path=img_path_input,
      num_of_examples=num_og_guesses,
    )
  else:
    answer = model.predict(
      img_path=img_path_input,
    )  
  print(f"image contains {answer}")
  
if __name__ == "__main__":
  main()
  
