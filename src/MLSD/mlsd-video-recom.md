
# Design a video recommendation system 

## 1. Problem Formulation 
User-video interaction 

Some existing data examples:  
* videos data
* User historic data 
* Recommendations data 
* Reviews 
<!-- * video features include:  
Url, 
types
name,
desc_snippet,
recent_reviews,
all_reviews,
release_date,
developer,
publisher,
popular_tag, -->

### Clarifying questions 
- Use case? Homepage?
  - Does user sends a text query as well?
- Business objective? 
  - Increase user engagement (play, like, click, share), purchase?, create a better ultimate gaming experience 
- Similar to previously played, or personalized for the user? Personalized for the user
- User locations? Worldwide (multiple languages) 
- User’s age group: 
- Do users have any favorite lists, play later, etc?
- How many videos? 100 million
- How many users? 100 million DAU
- Latency requirements  - 200msec?
- Data access 
  - Do we log and have access to any data? Can we build a dataset using user interactions ?
  - Do we have textual description of items? 
- can users become friends on the platform and do we wanna take that into account?
- Free or Paid?  




### ML objective

- Recommend most engaging (define) videos
  * Max. No. of clicks (clickbait)
  * Max. No. completed videos/sessions/levels (bias to shorter)
  * Max. total hours played ()
  * Max. No. of relevant items (proxy by user implicit/explicit reactions) -> more control over signals, not the above shortcomings

* Define relevance: e.g. like is relevant, or playing half of it is, …
* ML Objective: build dataset and model to predict the relevance score b/w user and a video
* I/O: I: user_id, O: ranked list of videos + relevance score
* ML category: Recommendation System

## 2. Metrics (Offline and Online) 

* Offline: 
    * precision @k, mAP, and diversity 
* Online: 
    * CTR, # of completed, # of purchased, total play time, total purchase, user feedback 

## 3. Architectural Components (MVP Logic) 
The main approaches used for personalized recommendation systems: 
* Content-based filtering: suggest items similar to those user found relevant (e.g. liked)
    * No need for interaction data, recommends new items to users (no item cold start)
    * Capture unique interests of users
    * New user cold start 
    * Needs domain knowledge 
* CF: Using user-user (user based CF) or item-item similarities (item based CF)
    * Pros
        * No domain knowledge 
        * Capture new areas of interest 
        * Faster than content (no content info needed)
    * Cons: 
        * Cold start problem (both user and item)
        * No niche interest 
* Hybrid 
    * Parallel hybrid: combine(CF results, content based)
    * Sequential: [CF based] -> Content based

What do we choose? 
We choose a sequential hybrid model (standard e.g. for video recommendation)

We follow  the three stage recommender system (funnel architecture) in order to meet latency requirements and eb able to scale the system to billions of items. 

```mermaid
   Candidate generation --> Ranking --> Re-ranking
```

In the first stage, we use a light model to retrive thousands of items from millions
In the second (ranking) stage, we focus on high precision using a powerful model. This will not impact serving speed much because it's only run on smaller subset of items. 

Candidate generation in practice comes from aggregation of different candidate generation models. Here we can assume three candidate generation modules: 

1. Candidate generation 1 (Relevance based)
2. Candidate generation 2 (Popularity)
3. Candidate generation 3 (Trending)  

where we use CF for candidate generation 1

We use content based modeling for ranking.

## 4. Data preparation 

Data Sources: 

1. Users (user profile, historical interactions):
     * User profile
       * User_id, username, age, gender, location (city, country), lang, timezone


2. videos (structures, metadata, video content - what is it?)
   - video_id, title, date, rating, expected_length?,  #reviews, language, tags, description, price, developer, publisher, level, #levels

3. User-video interactions:  
Historical interactions: Play, purchase, like, and search history, etc  
   - User_id, video_id, timestamp, interaction_type(purchase, play, like, impression, search), interaction_val, location


1. Context: time of the day, day of the week, device, OS

Type

- Removing duplicates 
- filling missing values 
- normalizing data.

### Labeling: 
For features in the form of <user, video> pairs -> labeling strategy based on explicit or implicit feedback 
e.g. "positive" if user liked the item explicitly or interacted (e.g. watched/played) at least for X (e.g. half of it).   
negative samples: sample from background distribution -> correct via importance smapling 

## 5. Feature engineering 

There are several machine learning features that can be extracted from videos. Here are some examples:

- video metadata features
- video state: e.g. the positions of players, the status of objects and obstacles, the time remaining, and the score.
- video mechanics: The rules and interactions that govern the video. 
- User engagement: e.g. the length of play sessions, frequency of play, and player retention rates.
- Social interactions: b/w players: to identify patterns of behavior, such as the formation of alliances, the sharing of resources, and the types of communication used between players.
- Player preferences: which video features are most popular among players, which can help inform video design decisions.
- Player behaviors: player movement patterns, the types of actions taken by players, and the strategies used to achieve objectives.


We select some important features as follows:

* video metadata features: 
  * video ID, 
  Duration, 
  Language, 
  Title, 
  Description,
  Genre/Category, 
  Tags,  
  Publisher(popularity, reviews), 
  Release date, 
  Ratings, 
  Reviews, 
  (video content ?)
video titles, genres, platforms, release dates, user ratings, and user reviews.



* User profile: 
  * User ID, Age, Gender, Language, City, Country 

* User-item historical features: 
  * User-item interactions 
    * Played, liked, impressions
    * purchase history (avg. price)
  * User search history 

* Context


### Feature representation: 

* Categorical data (video_id, user_id, language, city): Use embedding layers, learned during
training 
* Categorical_data(gender, age): one_hot
* Continuous variables: normalize, or bucketize and one-hot (e.g. price) 
* Text:(title, desc, tags): title/description use embeddings, pre-trained BERT, fine tune on video language?, tags: CBOW
* 
* video content embeddings? 

## 6. Model Development and Offline Evaluation 

### 6.1 Candidate Generation 

For candidate generation 1 (Relevance Based), we choose CF. 

For CF there are two embedding based modeling options: 
1. Matrix Factorization 
   * Pros: Training speed (only two matrices to learn), Serving speed (static learned embeddings)
   * Cons: only relies on user-item interactions (No user profile info e.g. language is used); new-user cold start problem 
2. Two tower neural network:
   * Pros: Accepts user features (user profile + user search history) -> better quality recommendation; handles new users 
   * Cons: Expensive training, serving speed     

We chose two-tower network here. 

#### Two-tower network
  * two encoder towers (user tower + encoder tower)
  * user tower encodes user features into user embeddings  $u$ 
  * item tower encodes item features into item embeddings   $v_i$
  * similarity $u$, $v_i$ is considered as a relevance score (ranking as classification problem)


#### Loss function: 
Minimize cross entropy for each positive label and sampled negative examples 

### 6.2 Ranking 
For Ranking stage, we prioritize precision over efficiency. We choose content based filtering. Choose a model that relies in item features.  
ML Obj options: 
   - max P(watch| U, C)
   - max expected total watch time 
   - multi-objective (multi-task learning: add corresponding losses)
  
Model Options: 
- FF NN (e.g. similar tower network to a tower network) + logistic regression 
- Deep Cross Network (DCN)

Features  

* Video ID embeddings (watched video embedding avg, impression video embedding), 
* Video historic
  *  No. of previous impressions, reviews, likes, etc
  *  Time features (e.g. time since last play), 
* Language embedding (user, item), 
* User profile 
* User Historic (e.g. search history)



### 6.3 Re-Ranking 
Re-ranks items by additional business criteria (filter, promote)  
We can use ML models for clickbait, harmful content, etc or use heuristics   
Examples: 
* Age restriction filter 
* Region restriction filter 
* Video freshness (promote fresh content)
* Deduplication 
* Fairness, bias, etc 




## 7. Prediction Service 
two-tower network inference: find the k-top most relevant items given a user ->  
It's a classic nearest neighbor problem -> use approximate nearest neighbor (ANN) algorithms  

## 8. Online Testing and Deployment  
Standard approaches as before.  
## 9. Scaling
The three stage candidate generation - ranking - re-ranking can be scaled well as described earlier. It also meets the requirements of speed (funnel architecture), precision(ranking component), and diversity (multiple candid generation). 

### Cold start problem: 
* new users: two tower architectures accepts new users and we can still use user profile info even with no interaction 
* new items: recommend to random users and collect some data - then fine tune the model using new data

### Training: 
We need to be able to fine tune the model 
### Exploration exploitation trade-off 
- Multi-armed bandit (an agent repeatedly selects an option and receives a reward/cost. The goal of to maximize its cumulative reward over time, while simultaneously learning which options are most valuable.)
### Other Extensions: 
* [Multi-task learning](https://daiwk.github.io/assets/youtube-multitask.pdf)
  * Includes a shared feature extractor that is trained jointly with multiple prediction heads, each of which is responsible for predicting a different aspect of user behavior, such as click-through rate, watch time, and view count. The model is trained using a combination of supervised and unsupervised learning techniques, including cross-entropy loss, pairwise ranking loss, and self-supervised contrastive learning.
* Positional bias (detection and correction) 
* Selection bias (detection and correction)
* Add negative feedback (dislike)
* Locality preservation: 
  * Use sequential user behavior info (CBOW model)
* effect of seasonality 
* what if we only have a query and personal (item, provider) history? 
  * item embeddings, provider embeddings, query embeddings 
  * we can build a query-aware attention mechanism that computes 

### More resources 

* [Content-based](https://www.kaggle.com/code/fetenbasak/content-based-recommendation-video-recommender), [NLP analysis](https://www.kaggle.com/code/greentearus/steam-reviews-nlp-analysis), [Collaborative Denoising AE](https://www.kaggle.com/code/krsnewwave/collaborative-denoising-autoencoder-steam)  
* [User-based CF, item-based CF and MF](https://github.com/manandesai/video-recommendation-engine) ([github](https://github.com/manandesai/video-recommendation-engine/blob/main/recommenders.ipynb))
* [CF and content based](https://github.com/AudreyGermain/video-Recommendation-System) 
