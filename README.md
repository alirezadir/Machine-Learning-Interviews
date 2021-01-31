# :orange_book: A Guide to Machine Learning Interviews :robot: :computer:

:label: Note: This repo is still under development, and all feedback and contribution are very welcome :blush:

This repo aims to be an enlightening guideline to prepare for **Machine Learning / AI interviews**. It has been developed based on my personal experience and notes during my own ML interview preparation early 2020 (right at the beginning of COVID19 :D), when I received offers from Facebook (ML Specialist), Google (ML Engineer), Amazon (Applied Scientist), Apple (Applied Scientist) , and Roku.

At the time I'm putting these notes together, machine learning interviews at different companies do not follow a unique structure unlike software engineering interviews. However, I found some of the components very similar to each other, although under different namings.

My preparation was focused mostly for *Machine Learning Engineer* (and Applied Scientist) roles at big companies. Although relevant roles such as "Data Dcience" or "ML research scientist" have different structure, some of the modules reviewed can be still useful. For more understanding about diff

The following components are the most common interview modules that I found for ML engineer roles at different companies. We will go through them one by one and share how one can prepare.

# 1. General Coding (Algorithms and Data Structures) Interview :computer:
As an ML engineer, you're first expected to have a good understanding of general software engineering concepts, and in particular, basic algorithms and data structure.

Depending on the company and seniority level, there is usually one or two rounds of general coding interviews. The general coding interview is very similar to SW engineer coding interviews, and one can get prepared for this one same as other SW engineering roles.

## Leetcode

At this time, [leetcode](https://leetcode.com/) is the most popular place to practice coding questions. I practived with around 350 problems, which were roughly distributed as **55% Medium, 35% Easy, and 15% Hard** problems. You can find some information on the questions that I practiced in [Ma Leet Sheet](https://docs.google.com/spreadsheets/d/1A8GIgCIn7gvnwE-ZBymI-4-5_ZxQfyeQu99N6f5gEGk/edit#gid=656844248) - Yea I tried to have a little bit fun with it here and there to make the pain easier to carry :D (I will write on my apporach to leetcode in future.)

## Educative.io

I was introduced to [educative.io](https://www.educative.io/) by a friend of mine, amd soon found it super useful in understanding the concepts of CS algorithms in more depth via their nice visualizations as well as categorizations.
In particular, I found the [Grokking the Coding Interview](https://www.educative.io/courses/grokking-the-coding-interview) pretty helpful in organizing my mind on approaching interview questions with similar patterns. And the [Grokking Dynamic Programming Patterns for Coding Interviews](https://www.educative.io/courses/grokking-dynamic-programming-patterns-for-coding-interviews) with a great categorization of DP patterns made tackling DP problems a piece of cake even though I wa initially scared!

## Interview Kickstart
As I had never taken an algorithms course before and this was my first preparation for coding interviews, I decided to invest a bit on myself and took [interview kickstart](https://www.interviewkickstart.com/)'s  technical interview prep course. In particular, my favorites were algorithms classes taught by [Omkar](https://www.linkedin.com/in/omkar-deshpande-2013588/), who dived deep in algorithms with his unique approach, the mock interviews, which well prepared my sklillsets for the main interviews. **Remember:** Interviewing is a skill and the more skillful you are, the better the resulst will be. Another part of the program that I learned a lot from (while many others ignored :D), was career coaching sessions with [Nick](https://www.linkedin.com/in/nickcam/).


# 2. ML Coding :robot:
ML coding module may or may not exist in particular companies interviews. The good news is that, there are a limited number of ML algorithms that candidates are expected to be able to code. The most common ones include:
- k-means clustering
- k-nearest neighbors
- Decision trees
- Perceptron, MLP
- Linear regression
- Logistic regression
- SVM
- Sampling
  - stratified sampling
  - uniform sampling
  - reservoir sampling
  - sampling multinomial distribution
  - random generator
- NLP algorithms (if that's your area of work)
  - bigrams
  - tf-idf

#### Sample codes for popular ML algorithms
- Python [TBD]


# 3. ML Depth
ML depth interviews typically aim to measure your depth level of knowledge in both theoretical and practical machine learning. Although this may sound scary at the beginning, this could be potentially one of the easiest rounds if you know well what you have worked on before. In other words, ML depth interviews typically focus on your previous ML related projects, but as deep as possible!

Typically these sessions start with going through one of your past projects (which depending on the company, it could be either your or the interviewer's choice). It generally starts as a high level discussion, and the interviewer gradually dives deeper in one or multiple aspects of the project, sometimes until you get stuck (so it's totally ok to get stuck, maybe just not too early!).

The best advice to prepare for this interview is to know the details of what you've worked on  before (really well), even if it's 5 years ago (and not be like that guy who once replied "hmm ... I worked on this 5 years ago and I don't recall the details :D " ).

## Example
TBD

# 4. ML Breadth/Fundamentals
As the name suggests, this interview is intended to evaluate your general knowledge of ML concepts both from theoretical and practical perspectives. Unlike ML depth interviews, the breadth interviews tend to follow a pretty similar structure and coverage amongst different interviewers and interviewees.

The best way to prepare for this interview is to review your notes from ML courses as well some high quality online courses and material. In particular, I found the following resources pretty helpful.

## Courses and review material:
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning) (you can also find the [lectures on Youtube](https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN) )
- [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects)
- [Udacity's deep learning nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101) or  [Coursera's Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) (for deep learning)


If you already know the concepts, teh following resources are pretty useful for a quick review of different concepts:
- [StatQuest Machine Learning videos](https://www.youtube.com/watch?v=Gv9_4yMHFhI&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)
- [StatQuest Statistics](https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9) (for statistics review - most useful for Data Science rols)
- [Machine Learning cheatsheets](https://ml-cheatsheet.readthedocs.io/en/latest/)
- [Chris Albon's ML falshcards](https://machinelearningflashcards.com/)

## Most important topics to cover for breadth
### 1. Classic ML Concepts

#### ML Algorithms' Categories
  - Supervised, unsupervised, and semi-supervised learning (with examples)
    - Classification vs regression vs clustering
  - Parametric vs non-parametric algorithms
  - Linear vs Nonlinear algorithms
#### Supervised learning
  - Linear Algorithms
    - Linear regression
      - least squares, residuals,  linear vs multivariate regression
    - Logistic regression
      - cost function (equation, code),  sigmoid function, cross entropy
    - Support Vector Machines
    - Linear discriminant analysis
  - Bayesian algorithms
    - Naive Bayes
    - Maximum a posteriori (MAP) estimation
    - Maximum Likelihood (ML) estimation

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
    - Logistic Loss fcn
    - Cross Entropy (formula)
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

#### Unsupervised learning
  - Clustering
    - k-means clustering
    - Hierarchical clustering
    - ?
  - Gaussian Mixture Models
  - Latent semantic analysis
  - Hidden Markov Models (HMMs)
    - ?? fnc and ??
  - Dimension reduction techniques
    - PCA
    - ICA
    - T-sne


#### Bias / Variance (Underfitting/Overfitting)
- Regularization techniques
  - L1/L2 (Lasso/Ridge)
#### Sampling
- sampling techniques
  - Uniform sampling
  - Reservoir sampling
  - Stratified sampling
#### Missing data
 - TBD
#### Time complexity of ML algorithms
-


### 2. Deep learning
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
- Attention
  - self-attention
- Transformer Architecture (in details, yes, no kidding! -- in an ideal world, I wouldn't answer those detailed questions to anyone except the authors and teammates, as either you've designed it or memorized it!)



## Statistical significance
- p-values
## Other:
    - outliers
    - similarity/dissimilarity metrics

# Machine Learning System Design
debugging
###
### Recommendation Systems
- examples
- collaboration filtering
    - chi chi based
    - Matrix factorization, LDA?
- Content based filtering
## ML domains
### Computer vision
- TBD
### NLP
- TBD
- Chatbots
  - [CMU lecture on chatbots](http://tts.speech.cs.cmu.edu/courses/11492/slides/chatbots_shrimai.pdf)
  - [CMU lecture on spoken dialogue systems](http://tts.speech.cs.cmu.edu/courses/11492/slides/sds_components.pdf)



# Machine Learning Design
## ML at Scale
other: chip h
## ML at Companies
- ML at LinkedIn
- discover, HST, Relevance
- analytics, xnlt for a/b testing
- feed ranking
- AI behind linkedin Feed
- Follow feed
- knowledge graph
-
- ML at Google
    - ML pipelines with TFX and KubeFlow
    - [How Google Search works](https://www.google.com/search/howsearchworks/)
      - Page Rank algorithm ([intro to page rank](https://www.youtube.com/watch?v=IKXvSKaI2Ko), [the algorithm that started google](https://www.youtube.com/watch?v=qxEkY8OScYY))
- Scalable ML using AWS
-  ML at Facebook
   -  [ML](https://www.youtube.com/watch?v=C4N1IZ1oZGw)
   -  [Scaling AI Experiences at Facebook with PyTorch](https://www.youtube.com/watch?v=O8t9xbAajbY)
   -  [Understanding text in images and videos](https://ai.facebook.com/blog/rosetta-understanding-text-in-images-and-videos-with-machine-learning/)
   -  [Protecting people](https://ai.facebook.com/blog/advances-in-content-understanding-self-supervision-to-protect-people/)
   -  Ads
   - Ad CTR prediction
     - [Practical Lessons from Predicting Clicks on Ads at Facebook](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)
     - Other [Ad papers](https://github.com/wzhe06/Ad-papers)
   - Newsfeed Ranking
     - [How Facebook News Feed Works](https://techcrunch.com/2016/09/06/ultimate-guide-to-the-news-feed/)
     - [How does Facebook’s advertising targeting algorithm work?](https://quantmar.com/99/How-does-facebooks-advertising-targeting-algorithm-work)
     - [ML and Auction Theory](https://www.youtube.com/watch?v=94s0yYECeR8)
     - [Serving Billions of Personalized News Feeds with AI - Meihong Wang](https://www.youtube.com/watch?v=wcVJZwO_py0&t=80s)
     - [Generating a Billion Personal News Feeds](https://www.youtube.com/watch?v=iXKR3HE-m8c&list=PLefpqz4O1tblTNAtKaSIOU8ecE6BATzdG&index=2)
     - Edgerank for news feed facebook
     - [Instagram feed ranking](https://www.facebook.com/atscaleevents/videos/1856120757994353/?v=1856120757994353)
     - [How Instagram Feed Works](https://techcrunch.com/2018/06/01/how-instagram-feed-works/)
   - [Photo search](https://engineering.fb.com/ml-applications/under-the-hood-photo-search/)
   - Fake news detection
   - Social graph search
   - Recommendation
     - [Recommending items to more than a billion people](https://engineering.fb.com/core-data/recommending-items-to-more-than-a-billion-people/)
     - [Social recommendations](https://engineering.fb.com/android/made-in-ny-the-engineering-behind-social-recommendations/)
   - [Live videos](https://engineering.fb.com/ios/under-the-hood-broadcasting-live-video-to-millions/)
   - [Large Scale Graph Partitioning](https://engineering.fb.com/core-data/large-scale-graph-partitioning-with-apache-giraph/)
   - [TAO: Facebook’s Distributed Data Store for the Social Graph](https://www.youtube.com/watch?time_continue=66&v=sNIvHttFjdI&feature=emb_logo) ([Paper](https://www.usenix.org/system/files/conference/atc13/atc13-bronson.pdf))
   - NLP
     - [NLP at Facebook](https://www.youtube.com/watch?v=ZcMvffdkSTE)
-  ML at Netflix
   -  [Recommendation at Netflix](https://www.slideshare.net/moustaki/recommending-for-the-world)
   -  [Past, Present & Future of Recommender Systems: An Industry Perspective](https://www.slideshare.net/justinbasilico/past-present-future-of-recommender-systems-an-industry-perspective)
   -  [Deep learning for recommender systems](https://www.slideshare.net/moustaki/deep-learning-for-recommender-systems-86752234)
   -  [Reliable ML at Netflix](https://www.slideshare.net/justinbasilico/making-netflix-machine-learning-algorithms-reliable)
   -  [ML at Netflix (Spark and GraphX)](https://www.slideshare.net/SessionsEvents/ehtsham-elahi-senior-research-engineer-personalization-science-and-engineering-group-at-netflix-at-mlconf-sea-50115?next_slideshow=1)
   -  [Recent Trends in Personalization](https://www.slideshare.net/justinbasilico/recent-trends-in-personalization-a-netflix-perspective)
   -  [Artwork Personalization @ Netflix](https://www.slideshare.net/justinbasilico/artwork-personalization-at-netflix)



# Behavioral interviews
- STAR method [TBD]
