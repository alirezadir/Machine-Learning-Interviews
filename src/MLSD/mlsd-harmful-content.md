# Harmful content detection on social media

### 1. Problem Formulation

* Integrity deals with: 
    * Harmful content (focus here)
    * Harmful act/actors  
* Goal: monitor posts, detect harmful content, and demote/remove 
* Examples harmful content categories: violence, nudity, hate speech 
* ML objective: predict if a post is harmful 
  * Input: Post (text, image, video) 
  * Output:  P(harmful) 
* Data: 500M posts / day (about 10K annotated)
* Latency: can vary for different categories 
* Able to explain the reason to the users (category) 
* support different languages? Yes 

### 2. Metrics  
- Offline 
  - F1 score, P/R-AUC
- Online 
  - harmful impressions, percentage of valid appeals, proactive rate, etc 
### 3. Architectural Components  
* Multimodal input (image, text, etc): 
  * Multimodal fusion techniques 
    * Early Fusion 
    * Late Fusion 
* Multi-Label classification 

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
    * Vectorization options: 
      * Statistical (BoW, TF-IDF)
      * ML based encoders (BERT)
    * We chose pre-trained ML based encoders (need semantics of the text)
    * We chose Multilingual Distilled (smaller, faster) version of BERT (need context), DistilmBERT 
  * Images/ Videos:   
    * Preprocessing: decoding, scaling, normalization
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
    * account features (No. of followers /following)
    * violation history (No of violations, No of user reports, profane words rate)

### 6. Model Development and Offline Evaluation

### 7. Prediction Service

### 8. Online Testing and Deployment  

### 9. Scaling, Monitoring, and Updates
