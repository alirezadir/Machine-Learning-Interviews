# <a name="ml-sys"></a>  Machine Learning System Design

## 1. [ML System Design Flow](#ml-sys-d-f)
## 2. [ML System Design Sample Questions](#2-ml-system-design-sample-questions)
## 3. [ML System Design Topics](#ml-sys-d-t)
## 4. [ML at Big Tech companies](#ml-sys-d-c)


### Designing ML systems for production

Deploying deep learning models in production can be challenging, and it is beyond training models with good performance. Several distinct components need to be designed and developed in order to deploy a production level deep learning system.

<p align="center">
<img src="../imgs/components.png" title="" width="90%" height="80%">
</p>


Approaching an ML system design problem follows a similar logical flow to the generic software system design. (
For more insight on general system design interview you can e.g. check out [Grokking the System Design Interview
](https://www.educative.io/courses/grokking-the-system-design-interview)
and [System design primer](https://github.com/donnemartin/system-design-primer).). However, there are certain components in the design of an ML based system that needs to be addressed and need special attention, as you will see below in ML System Design Flow.

### ML System Design Interview 

- In an ML system design interview you are exposed to open ended questions with no single correct answer.
- The goal of ML system design interview is evaluate your your ability to zoom out and design a production-level ML system that can be deployed as a service within a company's ML infrastructure.


# 1. ML System Design Flow <a name="ml-sys-d-f"></a>

In order to design a solid ML system for real world applications, it is important to follow a design flow.
I recommend using the following **9-step ML System Design Flow** to design SW system solutions for ML-relevant business problems both at work and during interviews:

<!-- <p align="center">
<img src="https://user-images.githubusercontent.com/5262877/219497742-f70eca2a-4338-4362-8a6a-ec83057a3230.png" title="" width="40%" height="40%">
</p> -->

<center>

**The 9-step ML System Design Flow:**
</center>

<center>

|   |    |  
|- |-------- | 
|Step 1| [Problem Formulation](#1-problem-formulation) |
|Step 2 |[Metrics (Offline and Online)](#2-metrics-offline-and-online) |
|Step 3 |[Architectural Components (MVP Logic)](#3-architectural-components-mvp-logic)|
|Step 4 |[Data Collection and Preparation](#4-data-collection-and-preparation) |
|Step 5 |[Feature Engineering](#5-feature-engineering) |
|Step 6 |[Model Development and Offline Evaluation](#6-model-development-and-offline-evaluation) |
|Step 7 |[Prediction Service](#7-prediction-service) |
|Step 8 |[Online Testing and Deployment](#8-online-testing-and-model-deployment)  |
|Step 9 |[Scaling, Monitoring, and Updates](#9-scaling-monitoring-and-updates) |
| | |
</center>

Note: Remember when using this design flow during an interview to be flexible. According to the needs of the interview or the interests of the interviewer, you may skip some of these components or spend more time for a deep dive in one or two components.  

## 1. Problem Formulation

- Clarifying questions
- Use case(s) and business goal
- Requirements
  - Scope (features needed), scale, and personalization
  - Performance: prediction latency, scale of prediction
  - Constraints
  - Data: sources and availability  
- Assumptions

- Translate an abstract problem into an ML problem
  - ML objective,
  - ML I/O,
  - ML category (e.g.  binary classification, multi-classification, unsupervised learning, etc)
- Do we need ML to solve this problem?
  - Trade off between impact and cost
    - Costs: Data collection, data annotation, compute
    - if Yes, we choose an ML system to design. If No, follow a general system design flow.  
    - Note: in an ML system design interview we can assume we need ML.

## 2. Metrics (Offline and Online)

- Offline metrics (e.g. classification, relevance metrics)  
  - Classification metrics
    - Precision, Recall, F1, AUC ROC, mAP, log-loss, etc
      - Imbalanced data
  - Retrieval and ranking metrics
    - Precision@k, Recall@k, MAP, MRR, NDCG
  - Regression metrics: MSE, MAE, 
  - Problem specific metrics
    - Language: BLEU, BLEURT, GLUE, ROUGE, etc 
    - ads: CPE, etc  
  - Latency
  - Computational cost (in particular for on-device)
- Online metrics
  - CTR
  - Task/session success/failure rate/times,
  - Engagement rate
  - Reciprocal rank of first click etc,
  - Conversion rate
  - Counter metrics: direct negative feedback (hide, report)
- Trade-offs b/w metrics

## 3. Architectural Components (MVP Logic)

- High level architecture and main components
  - Non-ML components:
    - user, app server, DBs, KGs, etc and their interactions
  - ML components:
    - Modeling modules (e.g. candidate generator, ranker, ect)
    - Train data generator  
    ... 
- Modular architecture design
    - Model 1 architecture  (e.g. candidate generation)
    - Model 2 architecture (e.g. ranker, filter)
    - ...



## 4. Data Collection and Preparation  

- Data needs
  - target variable
  - big actors in signals (e.g. users, items, etc)
  - type (e.g. image, text, video, etc) and volume
- Data Sources
  - availability and cost
  - implicit (logging), explicit (e.g. user survey)
- Data storage
- ML Data types
  - structured (numerical, categorical),
  - unstructured(e.g. image, text)
- Sampling
  - Non-probabilistic sampling
  - Probabilistic sampling methods
    - random, stratified, reservoir, importance sampling
- Labelling (for supervised)
  - Labeling methods
    - Natural labels (extracted from data e.g. clicks, likes, purchase, etc)
      - Missing negative labels (not clicking is not a negative label):
        - Negative sampling
    - Explicit user feedback
    - Human annotation (super costly, slow, privacy issues)
  - Handling lack of labels
  - Programmatic labeling methods (noisy, pros: cost, privacy, adaptive)
    - Semi-supervised methods (from an initial smaller set of labels e.g. perturbation based)
    - Weak supervision (encode heuristics e.g. keywords, regex, db, output of other ML models)
  - Transfer learning:
    - pre-train on cheap large data (e.g. GPT-3),
    - zero-shot or fine-tune for downstream task  
  - Active learning
  - Labeling cost and trade-offs
- Class imbalance
  - Resampling
  - weighted loss fcn
  - combining classes  
- Data augmentation
- Data generation
  - Data ingestion (offline, online)
  - Feature generation (next)
  - Feature transform
  - Label generation
  - Joiner

## 5. Feature Engineering

- Choosing features
  - Define big actors (e.g. user, item, document, query, ad, context),
  - Define actor specific features (current, historic)
    - Example text features: n-grams (uni,bi), intent, topic, frequency, length, embeddings  
    - Example user features: user profile, user history, user interests  
  - Define cross features (e.g. user-item, or query-document features)
    - Example query-document features: tf-idf
    - Example user-item features: user-video watch history, user search history, user-ad interactions(view, like)
  - Privacy constraints
- Feature representation
  - One hot encoding
  - Embeddings
    - e.g. for text, image, graphs, users (how), stores, etc
    - how to generate/learn?
    - pre-compute and store
  - Encoding categorical features (one hot, ordinal, count, etc)
  - Positional embeddings
- Preprocessing features 
  - Needed for unstructured data 
    - Text: Tokenize (Normalize, pre-tokenize, tokenizer model (ch/word/subword level), post-process (add special tokens))
    - Images: Resize, normalize
- Missing Values
- Scaling/Normalization
- Feature importance
- Featurizer (raw data -> features)

## 6. Model Development and Offline Evaluation

- Model selection (baseline)
  - Heuristics -> simple model -> more complex model -> ensemble of models
    - Pros and cons, and decision
      - Note: Always start as simple as possible (KISS) and iterate over
    <!-- - More on Model Selection (TODO) -->

  - Typical modeling choices: 
    - Logistic Regression 
    - Linear regression 
    - Decision trees
    - GBDT and RF 
    - SVM
    - Neural networks 
      - FF 
      - CNN
      - RNN 
      - Transformers
  
  - Popular NN architectures:
    - Two stage funnel architecture (candidate generation + ranking )
    - Tow-tower architecture
    - Wide and deep learning
    - Deep cross network
    - Multi-task learning
    - Transformers
      - Encoder, Decoder, Encoder-decoder
    - Knowledge distillation (student-teacher network)
    - NLP
      - BERT, T5, GPT
    - Computer Vision
         -Object detectors (single stage, two-stage)
      - Vision Transformer




- Data splits (train, dev, test)
  - Portions
  - Splitting time-correlated data (split by time)
    - seasonality, trend  
  - Data leakage:
    - scale after split,
    - use only train split for stats, scaling, and missing vals
- Model training 
  - Loss functions 
    - MSE, Binary/Categorical CE, MAE, Huber loss, Hinge loss, contrastive loss, etc
  - Optimizers
    - SGD, AdaGrad, RMSProp, Adam, etc
  - Model validation  
  - Debugging <!-- - More on Debugging (TODO) -->
  - Offline vs online training  

- Model offline evaluation
- Iterate over MVP model
    - Model Selection
    - Data augmentation
    - Model update frequency
- Model calibration

## 7. Prediction Service

- Data processing and verification
- Web app and serving system
- Prediction service
- Batch vs Online prediction
  - Batch: periodic, pre-computed and stored, retrieved as needed - high throughput
  - Online: predict as request arrives - low latency
  - Hybrid: e.g. Netflix: batch for titles, online for rows
- ML on the Edge (on-device AI)
  - Network connection/latency, privacy, cheap
  - Memory, compute power, energy constraints  
  - Model Compression
    - Quantization
    - Pruning
    - Knowledge distillation
    - Factorization

## 8. Online Testing and Model Deployment

- A/B Experiments
  - How to A/B test?
    - what portion of users?
    - control and test groups
    - null hypothesis
- Bandits
- Shadow deployment
- Canary release

## 9. Scaling, Monitoring, and Updates

- Scaling for increased demand (same as in distributed systems)
  - Scaling general SW system (distributed servers, load balancer, sharding, replication, caching, etc)
    - Train data / KB partitioning
  - Scaling ML system
    - [Distributed training]()
      - Data parallel DT
      - RPC based DT   
    - Data parallelism (for training)
    - Model parallelism (for inference)
    - Monitoring, failure tolerance, updating
- Monitoring:
  - Logging
    - Features, predictions, metrics, events
  - Monitoring metrics
    - SW system metrics
    - ML metrics (accuracy related, predictions, features)
      - Online and offline metric dashboards  
  - Monitoring data distribution shifts
    - Types: Covariate, label and concept shifts
    - Detection (stats, hypothesis testing)
    - Correction
- System failures
  - SW system failure
    - dependency, deployment, hardware, downtime
  - ML system failure
    - data distribution difference (test vs online)
    - feedback loops
    - edge cases  
    - data distribution changes
  - Alarms
    - failures (data pipeline, training, deployment), low metrics, etc
- Updates: Continual training
  - Model updates
    - train from scratch or a base model
    - how often? daily, weekly, monthly, etc
  - Auto update models 
  - Active learning 
  - Human in the loop ML  

## 10. Beyond MVP

- Edge cases (e.g. invalid/junk input)
- Iterations over the base design
- Bias in training data
- Freshness, Diversity



# 2. ML System Design Sample Questions <a name="ml-sys-d-q"></a>

### Design a:
* ### Recommendation System
  - Video recommendation (Netflix, Youtube)
  - Friend/follower recommendation (Facebook, Twitter)
  - Replacement product recommendation (Instacart)
  - Rental recommendation (Airbnb)
  - Place recommendation
- ### Newsfeed system (ranking)
- ### Search system (retrieval, ranking)
  - Google search
- ### Ads serving system (retrieval, ranking)
<!-- - Ads click prediction system (ranking) -->
- ### Named entity linking system (tagging, resolution)
- ### Harmful/spam/illegal content detection system (ads/email)
- ### Fraud detection system
- ### Autocompletion / Typeahead suggestion system
- ### Ride matching system
- ### Language identification system
- ### Chatbot system
- ### [Question answering system]()
- ### Proximity service / Yelp
- ### Food delivery time  
- ### Self-driving car (Perception, Prediction, Planning)
- ### Sentiment analysis system
- ### Healthcare diagnosis system


# 3. ML System Design Topics <a name="ml-sys-d-t"></a>

I observed there are certain sets of topics that are frequently brought up or can be used as part of the logic of the system. Here are some of the important ones:

### Recommendation Systems

- Candidate generation 
  - Collaborative Filtering (CF)
    - User based, item based
    - Matrix factorization
    - Two-tower approach
  - Content based filtering
- Ranking 
- Learning to rank (LTR)
  - point-wise (simplest), pairwise, list-wise 

### Search and Ranking (Ads, newsfeed, etc)

- Search systems 
  - Query search 
  - Visual search 
  - Video search 
  - Two stage model 
    - document selection
    - document ranking 
- Ranking 
  - Newsfeed ranking system
  - Ads ranking system 
  <!-- - Ranking by relevance -->
  - Ranking as classification 
  - Multi-stage ranking + blender + filter
  <!-- - Information Retrieval -->


### NLP

- Feature engineering 
  - Preprocessing (tokenization)
- Text Embeddings
  - Word2Vec, GloVe, Elmo, BERT
- NLP Tasks:
  - Text classification
    - Sentiment analysis
    - Topic modeling
  - Sequence tagging  
    - Named entity recognition
    - Part of speech tagging
      - POS HMM
      - Viterbi algorithm, beam search
  - Text generation 
    - Language modeling
      - N-grams vs deep learning models (trade-offs)
      <!-- - Out of vocabulary problem -->
      - Decoding

  - Sequence 2 Sequence models
    - Machine Translation
      - Seq2seq models, NMT, Transformers
  - Question Answering 
  - [Adv] Dialog and chatbots
      - [CMU lecture on chatbots](http://tts.speech.cs.cmu.edu/courses/11492/slides/chatbots_shrimai.pdf)
      - [CMU lecture on spoken dialogue systems](http://tts.speech.cs.cmu.edu/courses/11492/slides/sds_components.pdf)
  
- Speech Recognition Systems
    - Feature extraction, MFCCs
    - Acoustic modeling
      - HMMs for AM
      - CTC algorithm (advanced)

### Computer Vision

- Image classification
  - VGG, ResNET
- [Object detection](https://viso.ai/deep-learning/object-detection/) 
  - Two stage models (R-CNN, Fast R-CNN, Faster R-CNN)
  - One stage models  (YOLO, SSD)
  - [Vision Transformer (ViT)](https://viso.ai/deep-learning/vision-transformer-vit/)
  - NMS algorithm 
- Object Tracking
<!-- - Popular architectures (AlexNet, VGG, ResNET, R-CNN, YOLO) -->
 
### Graph problems
- People you may know 
<!-- ### Personalization -->


# 4. ML at big tech companies  <a name="ml-sys-d-c"></a>

Once you learn about the basics, I highly recommend checking out different companies blogs on ML systems. You can refer to some of those resources in the [ML at Companies](ml-comapnies.md) section.

# More resources 

- For more insight on different components above you can check out the following resources):
  - [Full Stack Deep Learning course](https://fall2019.fullstackdeeplearning.com/)
  - [Production Level Deep Learning](https://github.com/alirezadir/Production-Level-Deep-Learning)
  - [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design)
  - [Stanford course on ML system design](https://online.stanford.edu/courses/cs329s-machine-learning-systems-design)