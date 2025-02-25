import sys
from models_package.pipeline import CustomPipeline

# python -u "c:\Users\Omen\Desktop\internship\task-2\main.py" data/animals_test_img/dog.jpeg "This is a dog"

def main() -> None:
  args = sys.argv[1:]
  
  # check if the input is correct
  
  img_path_input, text_input = args[0], args[1]
  pipe = CustomPipeline()
  result = pipe.predict(
    img_path=img_path_input,
    text=text_input,
  )
  print('-' * 50)
  print(result)

if __name__ == '__main__':
  main()