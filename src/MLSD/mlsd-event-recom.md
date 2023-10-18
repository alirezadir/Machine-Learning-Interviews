
# Design an event recommendation system 

## 1. Problem Formulation 

* Clarifying questions 
  - Use case? 
    - event recommendation system similar to eventbrite's. 
  - What is the main Business objective? 
    - Increase ticket sales  
  - Does it need to be personalized for the user? Personalized for the user
  - User locations? Worldwide (multiple languages) 
  - User’s age group: 
  - How many users? 100 million DAU
  - How many events? 1M events / month 
  - Latency requirements  - 200msec?
  - Data access 
    - Do we log and have access to any data? Can we build a dataset using user interactions ?
    - Do we have textual description of items? 
    - Can we use location data (e.g. 3rd party API)? (events are location based)
  - Can users become friends on the platform? Do we wanna use friendships?
  - Can users invite friends? 
  - Can users RSVP or just register?
  - Free or Paid? Both 

* ML formulation 
  * ML Objective: Recommend most relevant (define) events to the users to maximize the number of registered events
  * ML category: Recommendation system (ranking approach)
    * rule based system 
    * embedding based (CF and content based)
    * Ranking problem (LTR)
      * pointwise, pairwise, listwise 
    * we choose pointwise LTR ranking formulation 
  * I/O: In: user_id, Out: ranked list of events + relevance score
    * Pointwise LTR classifier I/O: I: <user_id, event_id>, O: P(event register) (Binary classification)

## 2. Metrics (Offline and Online) 

* Offline: 
    * precision @k, recall @ k (not consider ranking quality)
    * MRR, mAP, nDCG (good, focus on first element, binary relevance, non-binary relevance) -> here event register binary relevance so use mAP  
   
* Online: 
    * CTR, conversion rate, bookmark/like rate, revenue lift  

## 3. Architectural Components (MVP Logic) 
* We two stage (funnel) architecture for 
  * candidate generation 
    * rule based event filtering (e.g. location, etc)
  * ranking formulation (pointwise LTR) binary classifier  

## 4. Data preparation 

* Data Sources: 
  1. Users (user profile, historical interactions)
  2. Events 
  3. User friendships 
  4. User-event interactions
  5. Context


*  Labeling: 

## 5. Feature engineering 

* Note: Event based recommendation is more challenging than movie/video: 
   * events are short lived -> not many historical interactions -> cold start (constant new item problem)
   * So we put more effort on feature engineering (many meaningful features)

* Features: 
  - User features 
    - age (one hot), gender (bucketize), event history  
 
  - Event features 
    - price, No of registered, 
    - time (event time, length, remained time)
    - location  (city, country, accessibility)
    - description
    - host (& popularity)
  
  - User Event features 
    - event price similarity 
    - event description similarity 
    - no. registered similarity 
    - same city, state, country
    - distance 
    - time similarity (event length, day, time of day)
  
  - Social features 
    - No./ ratio of friends going 
    - invited by friends (No)
    - hosted by friend (similarity)
  
  - context 
    - location, time  

* Feature preprocessing 
  - one hot (gender)
  - bucketize + one hot (age, distance, time)

* feature processing 
  * Batch (for static) vs Online (streaming, for dynamic) processing 
  * efficient feature computation (e.g. for location, distance)
  * improve: embedding learning - for users and events 

## 6. Model Development and Offline Evaluation 

* Model selection 
  * Binary classification problem: 
    * LR (nonlinear interactions)
    * GBDT (good for structured, not for continual learning)
    * NN (continual learning, expressive, nonlinear rels)
  * we can start with GBDT as a baseline and experiment improvements by NN (both good options)
* Dataset 
  * for each user and event pair, compute features, and label 1 if registered, 0 if not 
  * class imbalance 
    * resampling 
    * use focal loss or class-balanced loss 

## 7. Prediction Service 
* Candidate generation 
  * event filtering (millions to hundreds)
    * rule based (given a user, e.g. location, type, etc filters)
* Ranking 
  * compute scores for <usr, event> pairs, and sort 

## 8. Online Testing and Deployment  
Standard approaches as before.  

## 9. Scaling
