# my-c2l
Project for independent study


### Setup
Currently I'm using Poetry to manage project dependencies. However, I've also included a requirements.txt file for setup with other package managers.

Setup with Poetry 
```
git clone https://github.com/Shinobs11/my-c2l.git
cd my-c2l
poetry install
poetry shell
```
(Poetry shell is needed whenever a new shell is created to add the virtual environment to the PATH env variable. VScode has support for automatically running scripts with the virtual environment. I'd imagine some other IDEs may have the same feature.)


Setup with pip
```
git clone https://github.com/Shinobs11/my-c2l.git
cd my-c2l
pip install -r requirements.txt
```


### Basic Usage
The shell scripts I've been using to run the processes are listed below. You can modify these or make your own. 
```
pretrain.sh -- Trains a language model that will be used in generating triplets.
generateTriplets.sh -- Generates triplets for the contrastive learning model to use in training.
contrastiveLearning.sh -- Trains a language model using triplets generated from generateTriplets.sh
evaluate.sh -- Evaluates language models. 
```
Running each of these in sequence should demonstrate how contrastive learning impacts the accuracy of the language model. 

### Advanced Usage
As of right now, each of the processes in the workflow are started from the "main.py" file by passing in their corresponding arguments.
Maybe one day I'll whip up a quick gui using React or PyGame or whatever other thing I decide to use, but for right now the workflow can only be accessed via command line.

The main.py file is called with python and has several arguments, some of which only have an effect on some processes.
```
python3 main.py 
  --pretrain: Runs the pre-training process
    --dataset-name: Name of dataset in datasets folder.
    --batch-size: Number of elements to be batched together (Larger numbers require more memory)
    --epoch-num: Number of epochs to train for
    --use-pinned-memory: Whether or not to use pinned memory (Can help with performance in cases with slow memory or disk)

  --generate_triplets: Runs the triplet generating process
    --dataset-name
    --batch-size
    --use-pinned-memory
  
  --contrastive-train: Runs the contrastive learning process
    --dataset-name
    --batch-size
    --epoch-num
    --lambda-weight: Multiplier applied to the triplet margin loss. Default = 0.1
  
  --evaluate: Evaluates models for a given dataset
    --dataset-name
    --batch-size
    --use-cl-model: True if evaluating contrastive learning model, false otherwise. Default = False 
```




### Adding new datasets
As of right now, there's no interactive way to add new datasets. 
Follow the scripts in the "dataset_splitting" folder to create your own script.



### Future improvements


### Dataset
Using https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews for my dataset.
Considering using https://github.com/acmi-lab/counterfactually-augmented-data for my dataset insetad as that's what the original author used as well.

Labels used for sentiment classification:
[1, 0] for positive, [0, 1] for negative.