# Multimodal Video Search System 

### 1. Problem Formulation
* Clarifying questions
    - What is the primary (business) objective of the search system?
    - What are the specific use cases and scenarios where it will be applied?
    - What are the system requirements (such as response time, accuracy, scalability, and integration with existing systems or platforms)?
    - What is the expected scale of the system in terms of data and user interactions?
    - Is their any data available? What format? 
    - Can we use video metadata? Yes 
    - Personalized? not required 
    - How many languages needs to be supported?
    
* Use case(s) and business goal
  * Use case: user enters text query into search box, system shows the most relevant videos 
  * business goal: increase click through rate, watch time, etc.  
* Requirements
  * response time, accuracy, scalability (50M DAU)
* Constraints
  * budget limitations, hardware limitations, or legal and privacy constraints
* Data: sources and availability
  * Sources: videos (1B), text 
  * 10M pairs of <video, text_query>. Videos have metadata (title, description, tags) in text format 
* Assumptions
* ML formulation: 
  * ML Objective: retrieve videos that are relevant to a text query  
  * ML I/O: I: text query from a user, O: ranked list of relevant videos on a video sharing platform  
  * ML category: Visual search + Text Search systems 

   
### 2. Metrics  
- Offline
  - Precision@k, mAP, Recall@k, MRR 
  - we choose MRR (avg rank of first relevant element in results) due to the format of our eval data <video, text> pair 
- Online 
  - CTR: problem: doesn't track relevancy, click baits  
  - video completion rate: partially watched videos might still found relevant by user 
  - total watch time
  - we choose total watch time: good indicator of relevance 

### 3. Architectural Components  
Multimodal search (video, text) for video content from text query: 
- Visual search system 
  - Text query -> videos (based on similarity of text and visual content) 
  - Two tower embedding architecture (video and text_query encoders)
- Textual search system 
  - search for most similar titles, descs, and tags  w/ text query 
  - we can use Inverted Index (e.g. elastic search) for efficient full text search 
    - An inverted index is a data structure that maps terms (words) to the documents or locations where they appear, enabling efficient text-based document retrieval, commonly used in search engines.

### 4. Data Collection and Preparation
We use provided annotated data in the format of <video_id, query>. 
### 5. Feature Engineering
- Preprocessing unstructured data 
  - Text pre-processing : normalization, tokenization, token to ids
  - Video preprocessing: decode into frames -> sample -> resize -> scale, normalize, color correct 

### 6. Model Development and Offline Evaluation
* Model Selection  
  - Text encoders: 
    - Text -> Vector (Embeddings)  
    - Two approaches: 
      - Statistical (BoW, TF-IDF)
      - ML encoders (word2vec, transformer based e.g. BERT)  
    - We chose transformer based (BERT). 

  - Video encoders: 
    - Video-level
      - more expensive, but captures temporal understanding
      - Example: ViViT (Video Vision Transformer)
    - Frame-level (from sample frames and aggregate)
      - less expensive (training and serving speed, compute power) 
      - Example: ViT 


* Model Training   
  - contrastive learning (similar to visual search system). 

### 7. Prediction Service
Components: 
- Visual search from text query 
  - text -> preprocess -> encoder -> embedding 
  - videos are indexed by their encoded embeddings 
  - search: using approximate nearest neighbor search (ANN)
- Textual search
  - using Elasticsearch (full text / fuzzy search)
- Fusion  
  - re-rank based on weighted sum of rel scores 
  - re-rank using a model 
- Re-ranking 
  - business level logic and policies 

### 8. Online Testing and Deployment  

### 9. Scaling, Monitoring, and Updates
