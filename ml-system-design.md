# <a name="ml-sys"></a>  Machine Learning System Design

1. [ML System Design Flow](#ml-sys-d-f)
2. [ML System Design Sample Questions](#ml-sys-d-q)
3. [ML System Design Topics](#ml-sys-d-t)
4. [ML at big tech companies](#ml-sys-d-c)

### Designing ML systems for production
This is one of my favorite interviews in which you can shine bright and up-level your career. I'd like to mention the following important notes:

- Remember, the goal of ML system design interview is NOT to measure your deep and detailed knowledge of different ML algorithms, but your ability to zoom out and design a production-level ML system that can be deployed as a service within a company's ML infrastructure.

- Deploying deep learning models in production can be challenging, and it is beyond training models with good performance. Several distinct components need to be designed and developed in order to deploy a production level deep learning system.
<p align="center">
<img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/components.png" title="" width="60%" height="60%">
</p>

- For more insight on different components above you can check out the following resources):
  - [Full Stack Deep Learning course](https://fall2019.fullstackdeeplearning.com/)
  - [Production Level Deep Learning](https://github.com/alirezadir/Production-Level-Deep-Learning)
  - [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design)
  - [Stanford course on ML system design](https://online.stanford.edu/courses/cs329s-machine-learning-systems-design)



# 1. ML System Design Flow <a name="ml-sys-d-f"></a>
Approaching an ML system design problem follows a similar flow to the generic software system design.
For more insight on general system design interview you can e.g. check out [Grokking the System Design Interview
](https://www.educative.io/courses/grokking-the-system-design-interview)
and [System design primer](https://github.com/donnemartin/system-design-primer).



I developed the following design flow that worked pretty well during my own interviews:

<p align="center">
<img src="https://user-images.githubusercontent.com/5262877/219497742-f70eca2a-4338-4362-8a6a-ec83057a3230.png" title="" width="40%" height="40%">
</p>


## 1. Problem Formulation 
   - What does it mean? 
    - Translate an abstract problem into an ML problem (identify it e.g. as binary classification, multi-classification, unsupervised learning, etc)
   - Use cases 
   - Requirements
   - Assumptions 
   - Do we need ML to solve this problem? 
   -   Trade off between impact and cost
        -   Costs: Data collection, data annotation, compute 
        - if Yes, we choose an ML system to design. If No, follow a general system design flow.  

## 2. Metrics (Offline and Online)
  - Offline metrics  
    - Accuracy metrics (precision, recall, F1, AUC ROC, etc)
      - imbalanced data
    - Latency 
    - Problem specific metric (e.g. CTR)
    - Computational cost (in particular for on-device)
  - Online metrics 

## 3. MVP Logic and Architectural Components
   - Model based vs rule based logic 
        - Pros and cons, and decision 
          -  Note: Always start as simple as possible (KISS) and iterate over 
    - Propose a simple model (e.g. a binary logistic regression classifier)
     

## 4. Data Collection and Preperation 
  - Needs 
    - type (e.g. image, text, video, etc) and volume
  - Sources
      - availability and cost 
  - Sampling 
    - Nonprobablistic sampling   
    - Probabilistic sampling methods 
      - random, stratified, reservoir, importance sampling
  - Labelling (for supervised)
    - Labling methods
      - Natural labels (extracted from data e.g. clicks, likes, purchase, etc)   
      - Human annotation (super costly, slow, privacy issues)
     - Handliing lack of labels
      - Programmatic labeling methods (noisy, pros: cost, privacy, adaptive)
        - Semi-supervised methods (from an initial smaller set of labels e.g. perturbation based)
        - Weak supervision (encode heuristics e.g. keywords, regex, db, output of other ML models)
      - Transfer learning: 
        - pretrain on cheap large data (e.g. GPT-3), 
        - zero-shot or fine-tune for downstream task  
      - Active learning
    - Labeling cost and trade-offs
  - Data splits (train, dev, test)
    - Portions
    - Splitting time-correlated data (split by time)
    - How to chose a test set?
    - Data leackage: 
      - scale after split, 
      - use only train split for stats, scaling, and missing vals
  - Class imbalance 
  - Data augmentation 
 
## 5. Feature Engineering 
  - Choosing features
    - Define big actors (e.g. user, item, context), 
    - Define actor specific features (e.g. user specific features)
    - Define cross features (e.g. user-item features)
  - Feature representation
    - One hot encoding
    - Embeddings (for text, image, graphs, users, etc)
    - Encoding categorical features (one hot, ordinal, count, etc) 
    - Positional embeddings 
  - Missing Values 
  - Scaling/Normalization 
  - Feature importance 
    
## 6. Model Development, Training, and Offline Evaluation 
  - Model 1 architecture  
  - Model 2 architecture 
  - ...
  - Model training procedure 
  - Model offline evaluations 
  - Debugging 
  - Iterate over MVP model
    - Model Selection 
    - data augmentation 

## 7. Inference/Prediction Service (online/batch)
  - Data processing and verification 
  - Prediction serivce 
  - Serving infra
  - Web app 
  - Batch vs Online prediction 
  - ML on the Edge (on-device AI)
    - Model Compression 
      - Quantization 
      - Pruning 
      - Knowledge distillation 
      - Factorization 

## 8. Online Testing and Model Deployment 
- A/B Test 
  - How to A/B test? 
    - what portion of users?
    - control and test groups 
- Bandits 
- Shadow deployment 
- Canary release 


## 9. Scaling, Monitoring, and Updates 
  - Scaling for increased demand (same as in distributed systems)
    - Scaling web app and serving system 
    - Data partitioning 
    - Data parallelism (for training)
    - Model parallelism (for inference)
  - Monitoring: 
    - Data distribution shifts 
      - covariate, label and concept shifts 
      - Detection (stats, hypothesis testing)
      - Correction 
    - Monitoring metrics 
      - SW system metrics 
      - ML metrics (accuracy related, predictions, features) 
    - System failures 
      - SW system failure 
        - dependency, deployment, hardware, downtime    
      - ML system failure 
        - data distribution diff (test vs online) 
        - feedback loops 
        - edge cases  
        - data distribution changes 
  - Continual training 

# 2. ML System Design Sample Questions <a name="ml-sys-d-q"></a>
Design a:
* Recommendation System 
  *  Video recommendation (Netflix, Youtube) 
  *  Friend/follower recommendation (Facebook, Twitter)
  *  Replacement product recommendation (Instacart)
  *  Rental recommendation (Airbnb)
  *  Place recommendation 
* Newsfeed system (ranking)
* Search system (retrival, ranking)
  * Google saerch
* Ads click predicition system (ranking)
* Named entity tagging system 
* Spam/illegal ads detection system
* Fraud detection system 
* Autocompletion / Typeahead suggestion system 
* Ride matching system 
* Language identification system
* Chatbot system 
* Question ansering system
* Proximity service / Yelp
* Food delivery time  
* Self-driving car (Perception, Prediction, Planning)
* Sentiment analysis system 
* Healthcare diagnosis system 

More questions can be found [here](https://huyenchip.com/machine-learning-systems-design/exercises.html). 
 
# 3. ML System Design Topics <a name="ml-sys-d-t"></a>
I observed there are certain sets of topics that are frequently brought up or can be used as part of the logic of the system. Here are some of the important ones:

### Recommendation Systems
- Recommend the most relevant items to users 
- Collaborative Filtering (CF)
    - user based, item based
    - Cold start problem
    - Matrix factorization
- Content based filtering

### Ranking (Ads, newsfeed, etc)
- CTR prediction
- Ranking algorithms

### Information Retrieval
- Search
  - Pagerank
  - Autocomplete for search

### NLP
- Preprocessing
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
    - Seq2seq models, NMT, Transformers 


### Computer vision
- Image classification
- Object Tracking
- Popular architectures (AlexNet, VGG, ResNET)
- ...


# 4. ML at big tech companies  <a name="ml-sys-d-c"></a>
Once you learn about the basics, I highly recommend checking out different companies blogs on ML systems. You can refer to some of those resources in the [ML at Companies](ml-comapnies.md) section.


