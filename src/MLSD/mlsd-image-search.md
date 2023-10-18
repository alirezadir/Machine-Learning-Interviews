# Image Search System (Pinterest)

### 1. Problem Formulation
* Clarifying questions
    - What is the primary (business) objective of the visual search system?
    - What are the specific use cases and scenarios where it will be applied?
    - What are the system requirements (such as response time, accuracy, scalability, and integration with existing systems or platforms)?
    - How will users interact with the system? (click, like, share, etc)? Click only
    - What types of visual content will the system search through (images, videos, etc.)? Images only 
    - Are there any specific industries or domains where this system will be deployed (e.g., fashion, e-commerce, art, industrial inspection)?
    - What is the expected scale of the system in terms of data and user interactions?
    - Personalized? not required 
    - Can we use metadata? In general yes, here let's not. 
    - Can we assume the platform provides images which are safe? Yes
* Use case(s) and business goal
  * Use case: allowing users to search for visually similar items, given a query image by the user 
  * business goal: enhance user experience, increase click through rate, conversion rates, etc (depends on use case)
* Requirements
  * response time, accuracy, scalability (billions of images)
* Constraints
  * budget limitations, hardware limitations, or legal and privacy constraints
* Data: sources and availability
  * sources of visual data: user-generated, product catalogs, or public image databases?
  * Available? 
* Assumptions
* ML formulation: 
  * ML Objective: retrieve images that are similar to query image in terms of visual content 
  * ML I/O: I: a query image, and O: a ranked list of most similar images to the query image 
  * ML category: Ranking problem (rank a collection of items based on their relevance to a query)

### 2. Metrics  
* Offline metrics 
  * MRR 
  * Recall@k 
  * Precision@k 
  * mAP 
  * nDCG 
* Online metrics 
  * CTR 
  * Time spent on images 

### 3. Architectural Components  
* High level architecture 
  * Representation learning: 
    * transform input data into representations (embeddings) - similar images are close in their embedding space 
    * use distance between embeddings as a similarity measure between images 

### 4. Data Collection and Preparation
* Data Sources
  * User profile
  * Images 
    * image file
    * metadata
  *  User-image interactions: impressions, clicks: 
  * Context 
* Data storage
* ML Data types
* Labelling

### 5. Feature Engineering
* Feature selection 
  * User profile : User_id, username, age, gender, location (city, country), lang, timezone
  * Image metadata: ID, user ID, tags, upload date, ... 
  * User-image interactions: impressions, clicks: 
    * user id, Query img id, returned img id, interaction type (click, impression), time, location
* Feature representation 
  * Representation learning (embedding)
* Feature preprocessing 
  * common feature preprocessing for images: 
    * Resize (e.g. 224x224), Scale (0-1), normalize (mean 0, var 1), color mode (RGB, CMYK) 

### 6. Model Development and Offline Evaluation
* Model selection 
  * we choose NN because of 
    * unstructured data (images, text) -> NN good at it 
    * embeddings needed 
  * Architecture type: 
    * CNN based e.g. ResNet 
    * Transformer based (ViT)
    * Example: Image -> Convolutional layers -> FC layers -> embedding vector  
* Model Training 
  * contrastive learning -> used for image representation learning 
    * train to distinguish similar and dissimilar items (images)
* Dataset 
  * each data point: query img, positive sample (similar to q), n - 1 neg samples (dissimilar)
    * query img : randomly choose 
    * neg samples: randomly choose 
    * positive samples: human judge, interactions (e.g. click) as a proxy, artificial image generated from q (self supervision)
      * human: expensive, time consuming 
      * interactions: noisy and sparse 
      * artificial: augment (e.g. rotate) and use as a positive sample (similar to simCLR or MoCo) - data distribution differs in reality 
* Loss Function: contrastive loss 
  * contrastive loss: 
    * works on pairs (Eq, Ei)
    * calculate distance: b/w pairs -> softmax -> cross entropy <- Labels 
* Model eval and HP tuning 
* Iterations 
  
### 7. Prediction Service
* Prediction pipeline 

  * Embedding generation service 
    * image -> preprocess -> embedding gen (ML model) -> img embedding 
  * NN search service 
    * retrieve the most similar images from embedding space 
      * Exact: O(N.D)
      * Approximate(ANN) - sublinear e.g. O(D.logN)
        * Tree based ANN (e.g. R-trees, Kd-trees) 
          * partition space into two (or more) at each non-leaf node, 
          * only search the partition for query q 
        * Locality Sensitive Hashing LSH 
          * using hash functions to group points into buckets (close points into same buckets)
        * Clustering based 
    * We use ANN using an existing library like Faiss (Facebook)
  * Re-ranking service 
    * business level logic and policies (e.g. filter inappropriate or private items, deduplicate, etc)
* Indexing pipeline
  * Indexing service: indexes images by their embeddings 
  * keep the table updated for new images 
  * increases memory usage -> use optimization (vector / product quantization)

### 8. Online Testing and Deployment  
* A/B Test 
* Deployment and release 

### 9. Scaling, Monitoring, and Updates 
* Scaling (SW and ML systems)
* Monitoring 
* Updates 

### 10. Other points: 

