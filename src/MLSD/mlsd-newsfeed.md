# News Feed System 

### 1. Problem Formulation
show feed (recent posts and activities from other users) on a social network platform 
* Clarifying questions
  * What is the primary business objective of the system? (increase user engagement)
  * Do we show only posts or also activities from other users?
  * What types of engagement are available? (like, click, share, comment, hide, etc)? Which ones are we optimizing for? 
  * Do we display ads as well? 
  * What types of data do the posts include? (text, image, video)?
  * Are there specific user segments or contexts we should consider (e.g., user demographics)?
  * Do we have negative feedback features (such as hide ad, block, etc)?
  * What type of user-ad interaction data do we have access to can we use it for training our models? 
  * Do we need continual training? 
  * How do we collect negative samples? (not clicked, negative feedback). 
  * How fast the system needs to be? 
  * What is the scale of the system? 
  * Is personalization needed? Yes 
  
* Use case(s) and business goal
  * use case: show friends most engaging (and unseen) posts and activities on a social network platform app (personalized to user)
  * business objective: Maximize user engagement (as a set of interactions)

* Requirements;
    * Latency: 200 msec of newsfeed refreshed results after user opens/refreshes the app
    * Scalability: 5 B total users, 2 B DAU, refresh app twice 
    
* Constraints:
    * Privacy and compliance with data protection regulations.
    
* Data: Sources and Availability:
    * Data sources include user interaction logs, ad content data, user profiles, and contextual information.
    * Historical click and impression data for model training and evaluation.

* Assumptions:
    * Users' engagement behavior can be characterized by their explicit (e.g. like, click, share, comment, etc) or implicit interactions (e.g. dwell time) 
  
* ML Formulation:
    * Objective: 
      * maximize number of explicit, implicit, or both type of reactions (weighted)
      * implicit: more data, explicit: stronger signal, but less data -> weighted score of different interactions: share > comment > like > click etc 
    * I/O: I: user_id, O: ranked list of unseen posts sorted by engagement score (wighted sum) 
    * Category: Ranking problem: can be solved as pointwise LTR with multi/label (multi-task) classification

### 2. Metrics  
* Offline 
  * ROC AUC (trade-off b/w TPR and FPR)
* Online 
  * CTR, 
  * Reactions rate (like rate, comment rate, etc)
  * Time spent 
  * User satisfaction (survey)

### 3. Architectural Components  
* High level architecture 
  * We can use point-wise learning to rank (LTR) formulation 
  * Options for multi-label/task classification: 
    * Use N independent classifiers (expensive to train and maintain) 
    * Use a multi-task classifier
      * learn multi tasks simultaneously 
      * single shared layers (learns similarities between tasks) -> transformed features 
      * task specific layers: classification heads 
      * pros: single model, shared layers prevent redundancy, train data for each task can be used for others as well (limited data)

### 4. Data Collection and Preparation
* Data Sources
  * Users, 
  * Posts, 
  * User-post interaction 
  * User-user (friendship)

* Labelling

### 5. Feature Engineering

* Feature selection 
  * Posts: 
    * Text
    * Image/videos
    * No of reactions (likes, shares, replies, etc)
    * Age 
    * Hashtags 
  * User: 
    * ID, username
    * Demographics (Age, gender, location)
    * Context (device, time of day, etc)
    * Interaction history (e.g. user click rate, total clicks, likes, et )
  * User-Post interaction: 
    * IDs(user, Ad), interaction type, time, location 
  * User-user(post author) affinities 
    * connection type 
    * reaction history (No liked/commented/etc posts from author)

* Feature representation / preparation
  * Text: 
    * use a pre-trained LM to get embeddings
    * use BERT here (posts are in phrases usually, context aware helps) 
  
  * Image / Video: 
    * preprocess 
    * use pre-trained models e.g. SimCLR / CLIP to convert -> feature vector 
  
  * Dense numerical features: 
    * Engagement feats (No of clicks, etc)
      * use directly + scale the range
  * Discrete numerical: 
    * Age: bucketize into categorical then one hot 
  * Hashtags: 
    *  tokenize, token to ID, simple vectorization (TF-IDF or word2vec) - no context 


### 6. Model Development and Offline Evaluation

* Model selection 
  * We choose NN 
    * unstructured data (text, img, video)
    * embedding layers for categorical features
    * fine tune pre-trained models used for feat eng.
  * multi-labels 
    * P(click), P(like), P(Share), P(comment)
  * Two options: 
    * N NN classifiers  
    * Multi task NN (choose this) 
      * Shared layers 
      * Classification heads (click, like, share, comment)
  * Passive users problem: 
    * All their Ps will be small 
    * add two more heads 
      * Dwell time (seconds spent on post)
      * P(skip) (skip = spend time < t)
      

* Model Training 
  * Loss function: 
    * L = sum of L_is for each task 
    * for binary classif tasks: CE 
    * for regression task: MAE, MSE, or Huber loss
  * Dataset 
    * use features, post features, interactions, labels
    * labels: positive, negative for each task (like, didn't like etc)
      * for dwell time: it's a regression 
    * Imbalanced dataset: downsample negative 
  * Model eval and HP tuning 
  * Iterations 
  
### 7. Prediction Service
* Data Prep pipeline
  *  static features -> batch feature compute (daily, weekly) -> feature store
  *  dynamic features: # of post clicks, etc _> streaming  

* Prediction pipeline 
  * two stage (funnel) architecture 
    * candidate generation / retrieval service 
      * rule based 
      * filter and fetch unseen posts by users under certain criteria 
    * Ranking 
      * features -> model -> engagement prob. -> sort 
      * re-ranking: business logic, additional logic and filters (e.g. user interest category)
* Continual learning pipeline 
  * fine tune on new data, eval, and deploy if improves metrics  
  
### 8. Online Testing and Deployment  
* A/B Test 
* Deployment and release 

### 9. Scaling, Monitoring, and Updates 
* Scaling (SW and ML systems)
* Monitoring 
* Updates 

### 10. Other topics  
* Viral posts / Celebrities posts
* New users (cold start)
* Positional data bias 
* Update frequency 
* calibration: 
  * fine-tuning predicted probabilities to align them with actual click probabilities 
* data leakage: 
  * info from the test or eval dataset influences the training process
  * target leakage, data contamination (from test to train set)
* catastrophic forgetting 
  *  model trained on new data loses its ability to perform well on previously learned tasks 
