# multitask-training-evaluation
 A multitask training experiment that evaluates parameter sharing on a recommender system using Matrix Factorization and Neural Collaborative Filtering algorithms.
 
 Consider a set of users from a streaming platform such as Netflix or Youtube. At a high level, there are two loops:
Outer loop: In multitask learning, we frame each user as its own task, with its own dataset. 
Inner loop: For each user-movie pair, there are two tasks, predicting whether the user would watch the movie (factorization) and predicting the score that the user would rate the movie with (regression). 
The idea is to evaluate how parameter sharing and loss weighting affects performance.

## Running
1. install all dependencies.
2. Run tensorboard to visualize experiments.
```
$ tensorboard --logdir=run
```
Alternatively, run
```
$ sh run_all.sh
```
to recreate the experiment.

## Dataset
The movielens dataset was used. More information here: https://grouplens.org/datasets/movielens/
## Experiments
Four sets of parameters are compared, with different loss weighting and parameter sharing.
- Gray: parameter sharing enabled, factorization weight = 0.99, regression weight = 0.1.
- Blue: parameter sharing enabled, factorization weight = 0.5, regression weight = 0.5.
- Red: parameter sharing disabled, factorization weight = 0.5, regression weight = 0.5.
- Yellow: parameter sharing disabled, factorization weight = 0.99, regression weight = 0.1.

Factorization is the loss for the prediction task, 0 or 1, predicting whether the user would click on the recommendation.

Regression is the loss for the scoring task, predicting the user's rating for the recommendation.

Training metrics:
<img src="README_media/1.png">

Evaluation metrics:
<img src="README_media/2.png">
