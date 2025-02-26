# TASK 2 -- {Animal Classification, NER model} Pipeline

## setup instructions
1. Navigate to a task-2 directory.
2. Run "pip install -r requirements.txt" to install the required dependencies.
Now you can work with the Pipeline.

## usage

### setting-up models
#### classification model
train the model:
  python -m models_package.ner.train
the trained model'll be saved and ready to use in the pipeline

(optional) make some predictions:
  python -m models_package.ner.inference {image path}
the output will be the list of animals possibly depicted on that image. 
#### ner model
train the model:
  python -m models_package.ner.train
(optional) make some predictions:
  python -m models_package.ner.inference {text}
the text might look smth like "I see a dog on this picture"

Now you have the models are up and running.

### using pipeline

Run "python -u main.py {image path} {input text}" for the pipe to produce a boolean answer.
**example**:
  python -u main.py data/animals_test_img/dog.jpeg "This might be a dog"
The pipeline uses those saved models we trained earlier.

## note
### time spent
  three days (~7-9 hours each of them)

### impressions
  this task was harder. and i do mean it. i had but practical knowledge of comp. vision and nlp or transformer models. look at me know😄
  it was great though. hard. hard but great.  