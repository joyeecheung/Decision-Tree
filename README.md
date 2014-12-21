##Dependencies
1. Numpy
2. matplotlib
3. pillow

The source code can be run under windows or linux with python 2.7+ and the libraries above.

##Directory structure

    ./
    ├── doc
    │   └── report.pdf  (the report)
    ├── asset
    │   ├── dataset.txt  (data set)
    │   ├── learning-curve.png  (learning curve plot)
    │   ├── pruned-tree.png  (visualization of the pruned decision tree)
    │   └── tree.png  (visualization of the decision tree)
    └── src
        ├── main.py  (generate the learning curve)
        ├── preprocess.py  (split the dataset and store them in json)
        ├── tree.py  (for building, pruning, drawing decision trees)
        └── util.py  (directory structure configurations)


##How to generate the results

Note: python scripts should be run under the `src` directory. All images will be placed under the `asset` directory.

1. Make sure the data set `dataset.txt` is placed under `asset`
2. To sample from the dataset, run `python preprocess.py` under `src`. The training set and the test set will be stored as `training.json` and `testing.json` under `asset`. The default sampling probability of the test set is 0.5. If you need to change it, for example, to 0.3, run `python preprocess.py -p 0.3`.
3. To plot the decision tree built with `training.json` and `testing.json` generated with `preprocess.py`, run `python tree.py`. The plots will be placed under `asset` named `tree.png` and `pruned-tree.png`
4. To generate the learning curve, run `python main.py`. The plot will be placed under `asset` named `learning-curve.png`

##About
* [Github repository](https://github.com/joyeecheung/decision-tree.git)
* Time: Dec. 2014
