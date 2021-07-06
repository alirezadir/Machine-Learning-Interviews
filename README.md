[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Machine%20Learning%20Interview%20Guide&url=https://github.com/alirezadir/machine-learning-interview-enlightener&via=alirezadira&hashtags=artificialintelligence,machinelearning,interviews,design)
[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/fold_left.svg?style=social&label=Follow%20%40alirezadira)](https://twitter.com/alirezadira)

[![Build Status](https://travis-ci.com/username/projectname.svg?branch=master)](https://travis-ci.com/username/projectname)
[![codecov](https://codecov.io/gh/cam-barts/ObeyTheTestingGoat/branch/master/graph/badge.svg)](https://codecov.io/gh/cam-barts/ObeyTheTestingGoat)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/alirezadir/machine-learning-interview-enlightener/blob/main/LICENSE)

# :bulb: An Enlightening Guide to Machine Learning Interviews :orange_book: :computer: :robot: 

:label: Notes: 
* This repo is under continuous development, and any feedback and contribution are very welcome :blush: 
**If you'd like to contribute**, please make a pull request with your suggested changes). 

* This repo aims to be an enlightening guideline to prepare for **Machine Learning / AI technical interviews**. It has compiled based on my personal experience and notes from my own ML interview preparation early 2020, when I received offers from Facebook (ML Specialist), Google (ML Engineer), Amazon (Applied Scientist), Apple (Applied Scientist), and Roku.

* At the time I'm putting these notes together, machine learning interviews at different companies do not follow a unique structure unlike software engineering interviews. However, I found some of the components very similar to each other, although under different naming.

* My preparation was focused mostly for *Machine Learning Engineer* (and Applied Scientist) roles at big companies. Although relevant roles such as "Data Science" or "ML research scientist" have different structures, some of the modules reviewed here can be still useful. For more understanding about different technical roles within ML umbrella you can refer to [Link]

* As a supplementary resource, you can also refer to my [Production Level Deep Learning](https://github.com/alirezadir/Production-Level-Deep-Learning) repo for further insights on how to design deep learning systems for production. 

The following components are the most commonly used interview modules that I found for technical ML roles at different companies. We will go through them one by one and share how one can prepare:

1. [General Coding Interview (Algorithms and Data Structures)](#coding)
2. [ML/Data Coding](#ml-coding)
3. [ML Depth](#depth)
3. [ML Breadth](#breadth)
5. [Machine Learning System Design](#ml-sys)




# <a name="coding"></a> 1. General Coding Interview (Algorithms and Data Structures) :computer: 

As an ML engineer, you're first expected to have a good understanding of general software engineering concepts, and in particular, basic algorithms and data structure.

Depending on the company and seniority level, there are usually one or two rounds of general coding interviews. The general coding interview is very similar to SW engineer coding interviews, and one can get prepared for this one same as other SW engineering roles.

## Leetcode

At this time, [leetcode](https://leetcode.com/) is the most popular place to practice coding questions. I practiced with around 350 problems, which were roughly distributed as **55% Medium, 35% Easy, and 15% Hard** problems. You can find some information on the questions that I practiced in [Ma Leet Sheet](https://docs.google.com/spreadsheets/d/1A8GIgCIn7gvnwE-ZBymI-4-5_ZxQfyeQu99N6f5gEGk/edit#gid=656844248) - Yea I tried to have a little bit fun with it here and there to make the pain easier to carry :D (I will write on my approach to leetcode in the future.)

## Educative.io

I was introduced to [educative.io](https://www.educative.io/) by a friend of mine, and soon found it super useful in understanding the concepts of CS algorithms in more depth via their nice visualizations as well as categorizations.
In particular, I found the [Grokking the Coding Interview](https://www.educative.io/courses/grokking-the-coding-interview) pretty helpful in organizing my mind on approaching interview questions with similar patterns. And the [Grokking Dynamic Programming Patterns for Coding Interviews](https://www.educative.io/courses/grokking-dynamic-programming-patterns-for-coding-interviews) with a great categorization of DP patterns made tackling DP problems a piece of cake even though I was initially scared! Educative team released a new course for cracking the ML interviews w.r.t System Design [Grokking the Machine Learning Interview](https://www.educative.io/courses/grokking-the-machine-learning-interview). 

## Interview Kickstart
As I had never taken an algorithms course before and this was my first preparation for coding interviews, I decided to invest a bit in myself and took the [interview kickstart](https://www.interviewkickstart.com/)'s  technical interview prep course. In particular, my favorites were algorithms classes taught by [Omkar](https://www.linkedin.com/in/omkar-deshpande-2013588/), who dived deep in algorithms with his unique approach, the mock interviews, which well prepared my skill-sets for the main interviews. 

**Remember:** Interviewing is a skill and the more skillful you are, the better the results will be. Another part of the program that I learned a lot from (while many others ignored :D), was career coaching sessions.


# <a name="ml-coding"></a> 2. ML/Data Coding :robot:
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

## Sample codes
- You can find some sample codes (or links to ones) in the [ML_Coding_Problems Notebook](https://github.com/alirezadir/machine-learning-interview-enlightener/blob/main/Notebooks/ML_Coding_Problems.ipynb).


# <a name="depth"></a> 3. ML Depth
ML depth interviews typically aim to measure the depth of your knowledge in both theoretical and practical machine learning, in particular in the area that you claim you have worked on. Although this may sound scary at the beginning, this could be potentially one of the easiest rounds if you know well what you have worked on before. In other words, ML depth interviews typically focus on your previous ML related projects, but as deep as possible!

Typically these sessions start with going through one of your past projects (which depending on the company, it could be either your or the interviewer's choice). It generally starts as a high level discussion, and the interviewer gradually dives deeper in one or multiple aspects of the project, sometimes until you get stuck (so it's totally ok to get stuck, maybe just not too early!).

The best advice to prepare for this interview is to know the details of what you've worked on  before (really well), even if it's 5 years ago (and not be like that guy who once replied "hmm ... I worked on this 5 years ago and I don't recall the details :D " ).

**Examples:**

- [TBD]

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

# <a name="ml-sys"></a>  5. Machine Learning System Design

## Designing ML systems for production
This is one of my favorite interviews in which you can shine bright and up-level your career. I'd like to mention the following important notes:

- Remember, the goal of ML system design interview is NOT to measure your deep and detailed knowledge of different ML algorithms, but your ability to zoom out and design a production-level ML system that can be deployed as a service within a company's ML infrastructure.

- Deploying deep learning models in production can be challenging, and it is beyond training models with good performance. Several distinct components need to be designed and developed in order to deploy a production level deep learning system.
<p align="center">
<img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/components.png" title="" width="70%" height="70%">
</p>

- For more insight on different components above you can check out the following resources):
  - [Full Stack Deep Learning course](https://fall2019.fullstackdeeplearning.com/)
  - [Production Level Deep Learning](https://github.com/alirezadir/Production-Level-Deep-Learning)
  - [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design)
  - Stanford course on ML system design [TBA]

Once you learn about the basics, I highly recommend checking out different companies blogs on ML systems, which I learnt a lot from. You can refer to some of those resources in the subsection [ML at Companies](#ml-at-companies) below.

## ML System Design Flow
Approaching an ML system design problem follows a similar flow to the generic software system design.
For more insight on general system design interview you can e.g. check out:
- [Grokking the System Design Interview
](https://www.educative.io/courses/grokking-the-system-design-interview)
- [System design primer](https://github.com/donnemartin/system-design-primer)

Below is a design flow that I would recommend:

1. Problem Description
    - What does it mean? 
    - Use cases 
    - Requirements
    - Assumptions 
2. Do we need ML to solve this problem? 
    - Trade off between impact and cost
      - Costs: Data collection, data annotation, compute 
    - if Yes, go to the next topic. If No, follow a general system design flow. 
3. ML Metrics 
      - Accuracy metrics: 
          - imbalanced data?
      - Latency 
      - Problem specific metric (e.g. CTR)
4. Data
    - Needs 
        - type (e.g. image, text, video, etc) and volume
    - Sources
        - availability and cost 
    - Labelling (if needed)
      - labeling cost  
5. MVP Logic 
    - Model based vs rule based logic 
        - Pros and cons, and decision 
          -  Note: Always start as simple as possible and iterate over 
    - Propose a simple model (e.g. a binary logistic regression classifier)
    - Features/ Signals (if needed)
      - what to chose as and how to chose features 
      - feature representation 
6. Training (if needed)
      - data splits (train, dev, test)
        - portions
        - how to chose a test set 
      - debugging 
    - Iterate over MVP model (if needed)
      - data augmentation  
7. Inference (online)
    - Data processing and verification 
    - Prediction module 
    - Serving infra 
    - Web app 
8. Scaling
  - Scaling for increased demand (same as in distributed systems)
      - Scaling web app and serving system 
      - Data partitioning 
  - Data parallelism 
  - Model parallelism 
9. A/B test and deployment
    - How to A/B test? 
      - what portion of users?
      - control and test groups 
10. Monitoring and Updates 
    - seasonality   


## ML System Design Topics
I observed there are certain sets of topics that are frequently brought up or can be used as part of the logic of the system. Here are some of the important ones:

### Recommendation Systems
- Collaborative Filtering (CF)
    - user based, item based
    - Cold start problem
    - Matrix factorization
- Content based filtering

### NLP

- Preprocessing
  - Normalization, tokenization, stop words
- Word Embeddings
  - Word2Vec, GloVe, Elmo, BERT
- Text classification and sentiment analysis
- NLP specialist topics:
  - Language Modeling
  - Part of speech tagging
  - POS HMM
    - Viterbi algorithm and beam search
  - Named entity recognition
  - Topic modeling
  - Speech Recognition Systems
    - Feature extraction, MFCCs
    - Acoustic modeling
      - HMMs for AM
      - CTC algorithm (advanced)
    - Language modeling
      - N-grams vs deep learning models (trade-offs)
      - Out of vocabulary problem
  - Dialog and chatbots
    - [CMU lecture on chatbots](http://tts.speech.cs.cmu.edu/courses/11492/slides/chatbots_shrimai.pdf)
    - [CMU lecture on spoken dialogue systems](http://tts.speech.cs.cmu.edu/courses/11492/slides/sds_components.pdf)
  - Machine Translation
    - Seq2seq models, NMT

Note: The reason I have more topics here is because this was my focus in my own interviews

### Ads and Ranking
- CTR prediction
- Ranking algorithms

### Information retrieval
- Search
  - Pagerank
  - Autocomplete for search

### Computer vision
- Image classification
- Object Tracking
- Popular architectures (AlexNet, VGG, ResNET)
- [TBD]

### Transfer learning
- Why and when to use transfer learning
- How to do it
  - depending on the dataset sizes and similarities


## ML Systems at Big Companies 
- AI at LinkedIn
  - [Intro to AI at Linkedin](https://engineering.linkedin.com/blog/2018/10/an-introduction-to-ai-at-linkedin)
  - [Building The LinkedIn Knowledge Graph](https://engineering.linkedin.com/blog/2016/10/building-the-linkedin-knowledge-graph)
  - [The AI Behind LinkedIn Recruiter search and recommendation systems](https://engineering.linkedin.com/blog/2019/04/ai-behind-linkedin-recruiter-search-and-recommendation-systems)
  - [A closer look at the AI behind course recommendations on LinkedIn Learning, Part 1](https://engineering.linkedin.com/blog/2020/course-recommendations-ai-part-one)
  - [A closer look at the AI behind course recommendations on LinkedIn Learning, Part 2](https://engineering.linkedin.com/blog/2020/course-recommendations-ai-part-two)
  - [Communities AI: Building communities around interests on LinkedIn](https://engineering.linkedin.com/blog/2019/06/building-communities-around-interests)
  - [Linkedin's follow feed](https://engineering.linkedin.com/blog/2016/03/followfeed--linkedin-s-feed-made-faster-and-smarter)
  - XNLT for A/B testing


- ML at Google
    - ML pipelines with TFX and KubeFlow
    - [How Google Search works](https://www.google.com/search/howsearchworks/)
      - Page Rank algorithm ([intro to page rank](https://www.youtube.com/watch?v=IKXvSKaI2Ko), [the algorithm that started google](https://www.youtube.com/watch?v=qxEkY8OScYY))
    - TFX production components
      - [TFX workshop by Robert Crowe](https://conferences.oreilly.com/artificial-intelligence/ai-ca-2019/cdn.oreillystatic.com/en/assets/1/event/298/TFX_%20Production%20ML%20pipelines%20with%20TensorFlow%20Presentation.pdf)
    - [Google Cloud Platform Big Data and Machine Learning Fundamentals](https://www.coursera.org/learn/gcp-big-data-ml-fundamentals)
- Scalable ML using AWS
  - [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/)
  - [Deploy a machine learning model with AWS Elastic Beanstalk](https://medium.com/swlh/deploy-a-machine-learning-model-with-aws-elasticbeanstalk-dfcc47b6043e)
  - [Deploying Machine Learning Models as API using AWS](https://medium.com/towards-artificial-intelligence/deploying-machine-learning-models-as-api-using-aws-a25d05518084)
  - [Serverless Machine Learning On AWS Lambda](https://medium.com/swlh/how-to-deploy-your-scikit-learn-model-to-aws-44aabb0efcb4)
-  ML at Facebook
   -  [Machine Learning at Facebook Talk](https://www.youtube.com/watch?v=C4N1IZ1oZGw)
   -  [Scaling AI Experiences at Facebook with PyTorch](https://www.youtube.com/watch?v=O8t9xbAajbY)
   -  [Understanding text in images and videos](https://ai.facebook.com/blog/rosetta-understanding-text-in-images-and-videos-with-machine-learning/)
   -  [Protecting people](https://ai.facebook.com/blog/advances-in-content-understanding-self-supervision-to-protect-people/)
   -  Ads
      - Ad CTR prediction
      - [Practical Lessons from Predicting Clicks on Ads at Facebook](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)
   - Newsfeed Ranking
     - [How Facebook News Feed Works](https://techcrunch.com/2016/09/06/ultimate-guide-to-the-news-feed/)
     - [How does Facebook’s advertising targeting algorithm work?](https://quantmar.com/99/How-does-facebooks-advertising-targeting-algorithm-work)
     - [ML and Auction Theory](https://www.youtube.com/watch?v=94s0yYECeR8)
     - [Serving Billions of Personalized News Feeds with AI - Meihong Wang](https://www.youtube.com/watch?v=wcVJZwO_py0&t=80s)
     - [Generating a Billion Personal News Feeds](https://www.youtube.com/watch?v=iXKR3HE-m8c&list=PLefpqz4O1tblTNAtKaSIOU8ecE6BATzdG&index=2)
     - [Instagram feed ranking](https://www.facebook.com/atscaleevents/videos/1856120757994353/?v=1856120757994353)
     - [How Instagram Feed Works](https://techcrunch.com/2018/06/01/how-instagram-feed-works/)
   - [Photo search](https://engineering.fb.com/ml-applications/under-the-hood-photo-search/)
   - Social graph search
   - Recommendation
     - [Recommending items to more than a billion people](https://engineering.fb.com/core-data/recommending-items-to-more-than-a-billion-people/)
     - [Social recommendations](https://engineering.fb.com/android/made-in-ny-the-engineering-behind-social-recommendations/)
   - [Live videos](https://engineering.fb.com/ios/under-the-hood-broadcasting-live-video-to-millions/)
   - [Large Scale Graph Partitioning](https://engineering.fb.com/core-data/large-scale-graph-partitioning-with-apache-giraph/)
   - [TAO: Facebook’s Distributed Data Store for the Social Graph](https://www.youtube.com/watch?time_continue=66&v=sNIvHttFjdI&feature=emb_logo) ([Paper](https://www.usenix.org/system/files/conference/atc13/atc13-bronson.pdf))
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
