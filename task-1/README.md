# TASK 1 -- MNIST Classifier

## setup instructions
1. Navigate to a task-1 directory.
2. Run "pip install -r requirements.txt" to install the required dependencies.
Now you can work with the MnistClassifier. 

## usage
You may run "python -u main.py {model name} {training data source} {image path}" to predict a number, depicted on the provided image.
**example**:
  python -u main.py rf tensorflow data/my_digit_4.png

Those parameter are as follows:
1. {model name} may be of one of three types: 
  1) rf (Random Forest) - implemented with sklearn
  2) nn (Feed-Forward Neural Network) - with tf
  3) cnn (Convolutional Neural Network) - with tf as well
2. {training data source} has two options:
  1) tensorflow
  2) sklearn
and is responsible for the library fom which MNIST dataset will be downloaded. The dataset choice does not affect the model choice, e.g. the RF model (which is implemented with sklearn) works perfectly fine with a tensorflow dataset, due to a common DataProvider. 
3. {image path} path to an image to classify.

## note
### time spent
  one day (~8 hours)
### why both sklearn and tensorflow?
  I googled where to get the MNIST dataset from and found out that both of the above mentioned libs provide them. So I had a choice. But then, after I finished implementing a RF model with sklearn it struck me that NN and CNN would be written with TF. U can't have different models rely on different datasets. So there should be something that could provide a ds and that both sklearn- and tf- models could use. That's how DataProvider appeared.