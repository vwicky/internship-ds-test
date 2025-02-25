from models_package.mnist_classifier import MnistClassifier
import sys

def main() -> None:
  args = sys.argv[1:]
  
  # TODO: add input validation
  
  algorithm, provider, img_path = args[0], args[1], args[2]
  
  mnist_classifier = MnistClassifier(algorithm)  
  mnist_classifier.train(provider)
  mnist_classifier.evaluate()
  answer = mnist_classifier.predict(img_path=img_path)
  print(answer)

if __name__ == '__main__':
  main()