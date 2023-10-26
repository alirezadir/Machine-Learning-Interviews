# Search System 

### 1. Problem Formulation
* Clarifying questions
    - Is it a generalized search engine (like google) or specialized (like amazon product)?
    - What is the primary (business) objective of the search system?
    - What are the specific use cases and scenarios where it will be applied?
    - What are the system requirements (such as response time, accuracy, scalability, and integration with existing systems or platforms)?
    - What is the expected scale of the system in terms of data and user interactions?
    - Is their any data available? What format? 
    - Personalized? not required 
    - How many languages needs to be supported?
    - What types of items (products) are available on the platform, and what attributes are associated with them?
    - What are the common user search behaviors and patterns? Do users frequently use filters, sort options, or advanced search features?
    - Are there specific search-related challenges unique to the use case (e-commerce)? such as handling product availability, pricing, and customer reviews?

    
* Use case(s) and business goal
  * Use case: user enters text query into search box, system shows the most relevant items (products) 
  * business goal: increase CTR, conversion rate, etc  
* Requirements
  * response time, accuracy, scalability (50M DAU)
* Constraints
  * budget limitations, hardware limitations, or legal and privacy constraints
* Data: sources and availability
  * Sources:  
  * 
* Assumptions
* ML formulation: 
  * ML Objective: retrieve items that are most relevant to a text query  
    * we can define relevance as weighted summary of click, successful session, conversion, etc. 
  * ML I/O: I: text query from a user, O: ranked list of most relevant items on an e-commerce platform  
  * ML category: MM input search system -> retrieval and ranking 
    * ranking: MM input -> multi-label classification (click, success, convert, etc)
    * we can use a multi-task classifier 
   
### 2. Metrics  
- Offline
  - Precision@k, Recall@k, MRR, mAP, NDCG  
  - we choose NDCG (non-binary relevance)
- Online 
  - CTR: problem: doesn't track relevancy, click baits  
  - success session rate: dwell time > T or add to cart 
  - total dwell time 
  - conversion rate 

### 3. Architectural Components  
* Multimodal search (text, photo, video) for product content from text query: 
* Multi-layer architecture 
  * Query Understanding -> Candidate generation -> stage 1 Ranker -> stage 2 Ranker -> Blender -> Filter 
* Query understanding 
  * spell checker 
  * query normalization 
  * query expansion (e.g. add alternative) / relaxation (e.g. remove "good")
  * Intent/Domain classification 
* Candidate generation 
  * focus on recall, millions/billions into 10Ks 
* Ranking 
  * ML based 
  * multi-stage ranker: if more than 10k items to select from or QPS > 10k  
  * 100k items: stage 1 (liner model) -> stage 2 (DNN model) -> 500 items
* Blender: 
  * outputs a SERP (search engine result page)
  * blends results from multiple sources e.g. textual (inverted index, semantic) search, visual search, etc. 

#### Retrieval 
* from 100 B to 100k 
* IR: compares query text with document text 
* Document types: 
  * item (product) title 
  * item description 
  * item reviews 
  * item category 
* inverted index: 
  * index DS, mapping from words into their locations in a set of documents (e.g. ABC -> documents 1, 7)
* after query expansion (e.g. black pants into black and pants or suit-pants or trousers etc), do a search in inverted index db and find relevant items with relevance score 
* relevance score 
  * weighted linear combination of: 
    * terms match (e.g. TF-IDF score)(e.g. w = 0.5), 
    * item popularity (e.g. no of reviews, or bought) (e.g. w=0.125), 
    * intent match score (e.g. 0.125/2), 
    * domain match score,  
    * personalization score (e.g. age, gender, location, interests) 

#### Ranking: 
* see the next sections. 
<!-- 
- Visual search system 
  - Text query -> videos (based on similarity of text and visual content) 
  - Two tower embedding architecture (video and text_query encoders)
- Textual search system 
  - search for most similar titles, descs, and tags  w/ text query 
  - we can use Inverted Index (e.g. elastic search) for efficient full text search 
    - An inverted index is a data structure that maps terms (words) to the documents or locations where they appear, enabling efficient text-based document retrieval, commonly used in search engines. -->

### 4. Data Collection and Preparation
- Data sources: 
  - Users 
  - Queries 
  - Items (products)
  - Context 
- Labeling: 
  - use online user engagement data to generate positive and negative labels 
   
<!-- We use provided annotated data in the format of <video_id, query>.  -->
### 5. Feature Engineering
* Feature selection 
  * User: 
    * ID, username, 
    * Demographics (age, gender, location)
    * User interaction history (click rate, purchase rate, etc)
    * User interests (e.g. categories)
  * Context: 
    * device, 
    * time of the day, 
    * recent hype results 
    * previous queries 
  * Query features: 
    * query historical engagement (by other users)
    * query intent / domain 
    * query embeddings 
  * Item (product) features 
    * Title (exact text + embeddings)
    * Description (exact text + embeddings)
    * Reviews data (avg reviews, no of reviews, review textual data (text + embeddings)) 
    * category 
    * page rank 
    * engagement radius 
  * User-Item(product) features 
    * distance (e.g. for shipment)
    * historical engagement by the user (e.g. document type)
  * Query-Item(product) features
    * text match (title, description, category)
    * unigram or bigram search (title, description, category) - TF-IDF score 
    * historical engagement (e.g. click rate of Item for that query)
    * 
<!-- - Preprocessing unstructured data 
  - Text pre-processing : normalization, tokenization, token to ids
  - Video preprocessing: decode into frames -> sample -> resize -> scale, normalize, color correct  -->

### 6. Model Development and Offline Evaluation
#### Ranking 

* Model Selection  
  * Two options:
    * Pointwise LTR model: <user, item> -> relevance score 
      * approximate it as a binary classification problem p(relevant)
    * Pairwise LTR model: <user, item1, item2> -> item1 score > item2 score ?
      * loss function if the predicted order is correct 
      * more natural to ranking, more complicated 
  * Multi - Stage ranking 
    * 100k items (focus on recall) -> 500 items (focus on precision) -> 500 items in correct order   
    * Stage 1: We use a pointwise LTR -> binary classifier 
      * latency: microseconds 
      * suggestion: LR or small MART (multiple additive regression trees)
      * use ROC AUC for metric
    * Stage 2: Pairwise LTR model 
      * Two options (choose based on train data availability and capacity):
        * LambdaMART: a variation of MART, obj fcn changed to improve pairwise ranking  
        * LambdaRank: NN based model, pairwise loss (minimize inversions in ranking)
      * use NDCG for metric 

* Training Dataset
  * Pointwise approach 
    * positive samples: user engaged (e.g. click, spent time > T, add to cart, purchased)
    * negative samples: no engagement by the user + random negative samples e.g. from pages 10 and beyond
    * 5 million Q/day -> one positive one negative sample from each query -> 10 million samples a day 
    * use a whole week's data at least to capture daily patterns 
      * capturing and dealing with seasonal and holiday data 
    * train-valid/test split: 70/30 (of 70 million)
    * temporal affect: e.g. use 3 weeks data: first 2/3 of weeks: train, last week valid / test 
  * Pairwise approach: 
    * ranks items according to their relative order, which is closer to the nature of ranking 
    * predict doc scores in a way that miimizes No of inversions in the final ranked result 
    * Two options for train data generation for pointwise approach
      * human raters: each human rates 10 results per 100K queries * 10 humans = 10M examples
        * expensive, doesn't scale 
      * online engagement data 
        * assign scores to each engagement type e.g. 
          * impression with no click -> label/score 0 
          * click only -> score 1 
          * spent time after click > T : score 2 
          * add to cart : score 3 
          * purchase: score 4  
  
  <!-- - Text encoders:  -->
    <!-- - Text -> Vector (Embeddings)  
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
 -->

<!-- * Model Training   
  - contrastive learning (similar to visual search system).  -->

### 7. Prediction Service
<!-- - Visual search from text query 
  - text -> preprocess -> encoder -> embedding 
  - videos are indexed by their encoded embeddings 
  - search: using approximate nearest neighbor search (ANN)
- Textual search
  - using Elasticsearch (full text / fuzzy search)
- Fusion  
  - re-rank based on weighted sum of rel scores 
  - re-rank using a model 
- Re-ranking 
  - business level logic and policies  -->
- Re-ranking 
  - business level logic and policies  -->
    - filtering inappropriate items 
    - diversity (exploration/exploitation)
    - etc 
  - Two ways: 
    - rule based filters and aggregators 
    - ML model 
      - Binary Classification (P(inappropriate))
      - Data sources: human raters, user feedback (report, review)
      - Features: same as product features in ranker
      - Models: LR, MART, or DNN (depending on data size, capacity, experiments)
      - More details on harmful content classification 

### 8. Online Testing and Deployment  
### 9. Scaling, Monitoring, and Updates
### 10. Other talking points 
* Positional bias 