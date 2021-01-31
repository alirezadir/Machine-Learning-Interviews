# :orange_book: A Guide to Machine Learning Interviews :robot: :computer:

:label: Note: This repo is still under development, and all feedback and contribution are very welcome :blush:

This repo aims to be a guideline for preparing for machine learning interviews, and has been developed based on my personal experience and notes during my own interview prep. 

At the time I'm putting these notes together, machine learning interviews interviews at different companies do not follow a unique structure unlike software engineering interviews. However, I found some of the components very similar to each other, although under different namings. 

My preparation was focused mostly for "Machine Learning Engineer" (and Applied Scientist) roles at big companies. Although relevant roles such as "Data Dcience" or "ML reserachscientis" have different structure, some of the modules reviewed can be still useful. 

The following components are the most common interview modules that I found for ML engineer roles at different companies. We will go through them one by one and share how one can prepare. 

## 1. General Coding (Algorithms and Data Structures) Interview :computer:
As an ML engineer, you're first expected to have a good understanding of general software engineering concepts, and in particular, basic algorithms and data structure. 

Depending on the company and seniority level, there is usually one or two rounds of general coding interviews. The general coding interview is very similar to SW engineer coding interviews, and one can get prepared for this one same as other SW engineering roles. 

At this time, [leetcode](https://leetcode.com/) is the most popular place to practice coding questions. However, I also found [Grokking the Coding Interview](https://www.educative.io/courses/grokking-the-coding-interview) from [educative.io](https://www.educative.io/) pretty helpful in organizing my mind on approaching interview questions with similar patterns. 


## 2. ML Coding :robot:
ML coding module may or may not exist in particular companies interviews. The good news is that, there are a limited number of ML algorithms that candidates are expected to be able to code. The most common ones include:
- k-means clustering 
- k-nearest neighbors 
- Decision trees 


## ML Depth 

## ML Breadth/Fundamentals  
Courses and review material: 
- Andrew Ng's Machine Learning Course 
- Udacity 
- Machine Learning cheatsheets (only if you already know the concepts)
- 

### Classic ML Concepts 
Pros, cons
- ML Algorithms' Categories 
  - Supervised, unsupervised, and semi-supervised learning with examples 
  - Parametric vs non-parametric algorithms 
- ML Algorithms
- Linear and Logistic regression 
  - what is it
  - cost function, equation, code 
  - linear vs multivariate regression 
  - metrics for performance evaluation 
  - cost function 
  - Sigmoid function? cross entropy?
- Optimization 
  - Gradient descent and its variations 
- Support Vector Machines 
- Linear discriminant analysis  
- Bayesian algorithms 
  - Naive Bayes 
  - MAP
  - ML 
- Decision Trees
  - logits 
  - leaves?
  - training 
    - stop criteria 
  - inference
  - pruning 
- Ensemble methods 
  - Difference between bagging and boosting 
  - Random Forest 
  - Boosting
    - Adaboost 
    - GBM 
    - XGBoost 
  
  
- Clustering 
  - k-means clustering 
  - 
  - code
- Gaussian Mixture Models 
- Latent semantic analysis 
- HMMs  
- Dimension reduction techniques 
  - PCA
  - ICA 
  - T-sne 
## Deep learning 
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

## Bias/Variance (Underfitting/Overfitting)
- Regularization techniques 
  - L1/L2 (Lasso/Ridge)
## Sampling 
- sampling techniques 
  - Uniform sampling 
  - Reservoir sampling 
  - Stratified sampling 
## Missing data 

## Loss functions 
- Logistic Loss fcn
- Cross Entropy (formula)
- Hinge loss (SVM)
- 
## Feature selection + Model evaluation/selection 
    - Evaluation Metrics 
      - Accuracy, Precision, Recall 
      - F1 
      - ROC curve 
      - AUC 
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

## Machine Learning at Scale 
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

 
 
