# <a name="breadth"></a> 4. ML Breadth/Fundamentals
As the name suggests, this interview is intended to evaluate your general knowledge of ML concepts both from theoretical and practical perspectives. Unlike ML depth interviews, the breadth interviews tend to follow a pretty similar structure and coverage amongst different interviewers and interviewees.

The best way to prepare for this interview is to review your notes from ML courses as well some high quality online courses and material. In particular, I found the following resources pretty helpful.

## Courses and review material:
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning) (you can also find the [lectures on Youtube](https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN) )
- [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects)
- [Udacity's deep learning nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101) or  [Coursera's Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) (for deep learning)


If you already know the concepts, the following resources are pretty useful for a quick review of different concepts:
- [StatQuest Machine Learning videos](https://www.youtube.com/watch?v=Gv9_4yMHFhI&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)
- [StatQuest Statistics](https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9) (for statistics review - most useful for Data Science roles)
- [Machine Learning cheatsheets](https://ml-cheatsheet.readthedocs.io/en/latest/)
- [Chris Albon's ML falshcards](https://machinelearningflashcards.com/)

Below are the most important topics to cover:
## 1. Classic ML Concepts

### ML Algorithms' Categories
  - Supervised, unsupervised, and semi-supervised learning (with examples)
    - Classification vs regression vs clustering
  - Parametric vs non-parametric algorithms
  - Linear vs Nonlinear algorithms
### Supervised learning
  - Linear Algorithms
    - Linear regression
      - least squares, residuals,  linear vs multivariate regression
    - Logistic regression
      - cost function (equation, code),  sigmoid function, cross entropy
    - Support Vector Machines
    - Linear discriminant analysis

  - Decision Trees
    - Logits
    - Leaves
    - Training algorithm
      - stop criteria
    - Inference
    - Pruning

  - Ensemble methods
    - Bagging and boosting methods (with examples)
    - Random Forest
    - Boosting
      - Adaboost
      - GBM
      - XGBoost
  - Comparison of different algorithms
    - [TBD: LinkedIn lecture]

  - Optimization
    - Gradient descent (concept, formula, code)
    - Other variations of gradient descent
      - SGD
      - Momentum
      - RMSprop
      - ADAM
  - Loss functions
    - Logistic Loss function 
    - Cross Entropy (remember formula as well)
    - Hinge loss (SVM)

- Feature selection
  - Feature importance
- Model evaluation and selection
  - Evaluation metrics
    - TP, FP, TN, FN
    - Confusion matrix
    - Accuracy, precision, recall/sensitivity, specificity, F-score
      - how do you choose among these? (imbalanced datasets)
      - precision vs TPR (why precision)
    - ROC curve (TPR vs FPR, threshold selection)
    - AUC (model comparison)
    - Extension of the above to multi-class (n-ary) classification
    - algorithm specific metrics [TBD]
  - Model selection
    - Cross validation
      - k-fold cross validation (what's a good k value?)

### Unsupervised learning
  - Clustering
    - Centroid models: k-means clustering
    - Connectivity models: Hierarchical clustering
    - Density models: DBSCAN
  - Gaussian Mixture Models
  - Latent semantic analysis
  - Hidden Markov Models (HMMs)
    - Markov processes
    - Transition probability and emission probability
    - Viterbi algorithm [Advanced]
  - Dimension reduction techniques
    - Principal Component Analysis (PCA)
    - Independent Component Analysis (ICA)
    - T-sne


### Bias / Variance (Underfitting/Overfitting)
- Regularization techniques
  - L1/L2 (Lasso/Ridge)
### Sampling
- sampling techniques
  - Uniform sampling
  - Reservoir sampling
  - Stratified sampling
### Missing data
 - [TBD]
### Time complexity of ML algorithms
- [TBD]

## 2. Deep learning
- Feedforward NNs
  - In depth knowledge of how they work
  - [EX] activation function for classes that are not mutually exclusive
- RNN
  - backpropagation through time (BPTT)
  - vanishing/exploding gradient problem
- LSTM
  - vanishing/exploding gradient problem
  -  gradient?
- Dropout
  - how to apply dropout to LSTM?
- Seq2seq models
- Attention
  - self-attention
- Transformer and its architecture (in details, yes, no kidding! I was asked twice! In an ideal world, I wouldn't answer those detailed questions to anyone except the authors and teammates, as either you've designed it or memorized it!)
- Embeddings (word embeddings)


## 3. Statistical ML
###  Bayesian algorithms
  - Naive Bayes
  - Maximum a posteriori (MAP) estimation
  - Maximum Likelihood (ML) estimation
### Statistical significance
- R-squared
- P-values

## 4. Other topics:
  - Outliers
  - Similarity/dissimilarity metrics
    - Euclidean, Manhattan, Cosine, Mahalanobis (advanced)