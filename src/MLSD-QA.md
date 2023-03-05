# Design a review based QA System 
## 1. Problem Formulation 
## 2. Architectural Components (MVP Logic) 

## 3. Metrics (Offline and Online) 
- Offline 
    - Retriever: recall (per product, then aggregate over all)
    - Reader: Exact Match(EM) - after normalization, F1 Score 


## 4. Data Collection and Preparation 
## 5. Feature Engineering 
## 6. Model Development and Offline Evaluation 
## 7. Prediction Service 
## 8. Online Testing and Deployment  
## 9. Scaling, Monitoring, and Updates 

- business goal: help customers answer specific questions to evaluate a product  
- goal is to build a review-based QA system 
* Forms of QA: 
    - extractive QA: two stage architecture: 
        - retrieving relevant docs, 
        - extract answers from the the docs 
    - Community QA, 
    - long-form QA 
    - QA over tables 

1. option one, post question to the community QA like in Amazon. Cons: takes days to get an answer 
2. Buuid a system that automatically extracts an answer(s) from a collection of reviews  
Given product_id, Review(s), and a Question -> return the answer(s)

Data 
* QA systems: 
    - closed domain 
    - open domain  
* common format: 
    - (question, review, [answer sentences]) - format of SQuAD 

* Example dataset: SubjQA (> 1000 customer reviews in English, 6 domains)
    - QAs are subjective (more difficult than factual QAs)
    - important parts of query do not appear in review -> it can't be answered w/ keyword search or paraphrasing the query 
    - Cons: for each domain only 1-2k examples (reason: annotation super costly!)
* Other datasets: SQuAD 2.0 (augmented SQuAD 1.1 with adversarial questions that are relevant but can't be answered from the text! )

ML Problem Forumlation: 

The supervised learning problem can be framed as *Span Classification* task, where start and end of an answer span act as a label pair that model needs to predict for the classification task. I: (question, context) pair, O: start logits, end logits 

Note: our training set is small -> start with a pre-trained LM that has been fine tuned on a large scale QA system e.g. SQuAD. Use reading comprehension capability of this model as the *baseline*. 

Model selection: 
* Example models that perform well on SQuAD 2.0: 
    - RoBERTa, ALBERT, XLM-RoBERTa

<!-- TODO Add a figure here -->

Feature Engineering: 

Tokenizer 

Both question and context are passed as inputs. for each QA example, 
so we tokenize both as: 
    input_id, token_type_id(Q/A), attn_mask
the input takes the format (decoded version):

[CLS] question tokens [SEP] context tokens [SEP] (similar to BERT tokenization)


Model architecture: 
The retriever-reader architecture is similar to the tow stage funnel architecture (candidate generation + ranking) as seen before. 
<!-- TODO Add a figure here -->


1. Retriever:   
Select (retrieve) relevant passages from all the reviews in our system for a given query.  
categorized as 
    - Sparse (using frequencies, represent document and query as parse features) or 
        - e.g. based on TFIDF, BM25 (improved version of TF-IDF)
        (BM25 (Best Matching 25) is a ranking function used by search engines to score and rank documents based on their relevance to a user's query)
    - Dense retrievers (use encoders like transformers; represent document and query as contextual embeddings; encode semantic meaning).




- Dense retriever: 

2. Reader 

    Assuming that we have both question and relevant context (reviews)
    It's a reading comprehension model. 
    
    1. Choose a transformer encoder based model (e.g. variants of BERT such as RoBERTa, or ALBERT, or mini), 
    2. add a QA head (linear layer that takes hidden states from the encoder and computes logits for the start and end spans). 

<!-- TODO Add a figure here -->
- Example library that supports a retriever-reader architecture with transformers: Haystack from deepset.ai. 

Document Store  
A document oriented db to store docs and metadata 
Example options: Elasticsearch (supports both sparse and dense) and FAISS (only dense). 

We use Elasticsearch: can handle different types of data (text, numerical, geo-spatial, structured, etc) + huge data volume store with quick filter features in full-text search -> good option for a QA system 


Decoding: 
argmax over the start and end logits -> slice the span from the inputs to decode into the final answer 

- Note: we need to take care of Q's where there's no answer for (empty list) -> in the decoding make sure to convert them to empty strings 
- Note: We can have our model predict multiple answers by returning top-k 
- Dealing with long languages: context in QA often is longer than max seq. length in transformers (e.g. 512 tokens in MiniLM). This is ok with classification tasks but for QA is problematic as the answer might be close to the end of the context. Solution: Use a sliding window of max_len and stride. 

 
## Extensions 
 Domain Adaptation 