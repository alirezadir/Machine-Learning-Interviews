# Harmful content detection on social media

### 1. Problem Formulation
* Clarifying questions
  * What types of harmful content are we aiming to detect? (e.g., hate speech, explicit images, cyberbullying)?
  * What are the potential sources of harmful content? (e.g., social media, user-generated content platforms)
  * Are there specific legal or ethical considerations for content moderation
  * What is the expected volume of content to be analyzed daily?
  * What are supported languages? 
  * Are there human annotators available for labeling? 
  * Is there a feature for users to report harmful content? (click, text, etc). 
  * Is explainablity important here? 
  
* Integrity deals with: 
    * Harmful content (focus here)
    * Harmful act/actors  
* Goal: monitor posts, detect harmful content, and demote/remove 
* Examples harmful content categories: violence, nudity, hate speech 
* ML objective: predict if a post is harmful 
  * Input: Post (MM: text, image, video) 
  * Output:  P(harmful) or P(violent), P(nude), P(hate), etc
* ML Category: Multimodal (Multi-label) classification 
* Data: 500M posts / day (about 10K annotated)
* Latency: can vary for different categories 
* Able to explain the reason to the users (category) 
* support different languages? Yes 

### 2. Metrics  
- Offline 
  - F1 score, PR-AUC, ROC-AUC 
- Online 
  - prevalence (percentage of harmful posts didn't prevent over all posts), harmful impressions, percentage of valid (reversed) appeals, proactive rate (ratio of system detected over system + user detected) 

### 3. Architectural Components  
* Multimodal input (text, image, video, etc): 
  * Multimodal fusion techniques 
    * Early Fusion: modalities combined first, then make a single prediction 
    * Late Fusion: process modalities independently, fuse predictions
      * cons: separate training data for modalities, comb of individually safe content might be harmful 
* Multi-Label/Multi-Task classification 
  * Single binary classifier (P(harmful))
    * easy, not explainable 
  * One binary classifier per harm category (p(violence), p(nude), p(hate))
    * multiple models, trained and maintained separately, expensive 
  * Single multi-label classifier 
    * complicated task to learn 
  * Multi-task classifier: learn multi tasks simultanously 
    * single shared layers (learns similarities between tasks) -> transformed features 
    * task specific layers: classification heads 
    * pros: single model, shared layers prevent redundancy, train data for each task can be used for others as well (limited data)

### 4. Data Collection and Preparation

* Main actors for which data is available: 
  * Users 
    * user_id, age, gender, location, contact
  * Items(Posts) 
    * post_id, author_id, text context, images, videos, links, timestamp
  * User-post interactions 
    * user_id, post_id, interaction_type, value, timestamp


### 5. Feature Engineering
Features: 
Post Content (text, image, video) + Post Interactions (text + structured) + Author info + Context  
* Posts 
  * Text:  
    * Preprocessing (normalization + tokenization) 
    * Encoding (Vectorization): 
      * Statistical (BoW, TF-IDF)
      * ML based encoders (BERT)
    * We chose pre-trained ML based encoders (need semantics of the text)
    * We chose Multilingual Distilled (smaller, faster) version of BERT (need context), DistilmBERT 
  * Images/ Videos:   
    * Preprocessing: decoding, resize, scaling, normalization
    * Feature extraction: pre-trained feature extractors 
      * Images: 
        * CLIP's visual encoder 
        * SImCLR 
      * Videos: 
        * VideoMoCo
* Post interactions: 
  * No. of likes, comments, shares, reports (scale) 
  * Comments (text): similar to the post text (aggregate embeddings over comments)
* Users: 
  * Only use post author's info
    * demographics (age, gender, location)
    * account features (No. of followers /following, account age)
    * violation history (No of violations, No of user reports, profane words rate)
* Context: 
  * Time of day, device

### 6. Model Development and Offline Evaluation
* Model selection 
  * NN: we use NN as it's commonly used for multi-task learning 
* HP tuniing: 
  * No of hidden layers, neurons in layers, act. fcns, learning rate, etc
  * grid search commonly used 
* Dataset: 
  * Natural labeling (user reports) - speed 
  * Hand labeling (human contractors) - accuracy 
  * we use natural labeling for train set (speed) and manual for eval set (accuracy)
* loss function: 
  * L = L1 + L2 + L3 ... for each task 
  * each task is a binary classific so e.g. CE for each task  
* Challenge for MM training: 
  * overfitting (when one modality e.g. image dominates training)
    * gradient blending and focal loss 

### 7. Prediction Service
* 3 main components: 
  * Harmful content detection service 
  * Demoting service (prob of harm with low confidence)
  * violation service (prob of harm with high confidence)

### 8. Online Testing and Deployment  

### 9. Scaling, Monitoring, and Updates

### 10. Other topics 
* biases by human labeling 
* use temporal information (e.g. sequence of actions)
* detect fake accounts 
* architecture improvement: linear transformers 
