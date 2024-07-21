# Binary classification workflow

The code in this repository trains a binary classification model on a generated data set and then runs standard classification metrics, [SHAP](https://arxiv.org/abs/1705.07874), and [supervised clustering](https://arxiv.org/abs/1706.06060) to evaluate and interpret the results.

As the data set is synthetically generated, the purpose here is not to glean any particular insights about the data set or even about the utility of the modeling approach.  The purpose is to store code that represents a fairly typical modeling workflow that I might use and that can be readily adapted to a particular problem.

Generally speaking, the steps of the workflow are:

- Generate multivariate Gaussian-distributed data with correlations among the variables
- Dichotomize the response variable based on several different thresholds
- For each version of the response variable:
    - Split the data into training and test sets
    - Choose a modeling algorithm (currently [histogram-based gradient-boosted trees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html))
    - Define a hyperparameter space for the modeling algorithm
    - Use random search and cross-validation to optimize the hyperparamters on the training data set
    - Refit the modeling algorithm with the optimized hyperparameters to the entire training data set
    - Choose the classification threshold that maximizes the chosen performance metric for the training and test data sets
    - Calculate and plot standard binary classification metrics (e.g., confusion matrix, [ROC curve](https://en.wikipedia.org/wiki/Roc_curve))
    - Calculate and plot feature importances based on [SHAP](https://arxiv.org/abs/1705.07874) 
    - Calculate and plot [supervised clusters](https://arxiv.org/abs/1706.06060)

