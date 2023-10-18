# Offline Metrics 

These offline metrics are commonly used in search, information retrieval, and recommendation systems to evaluate the quality of results or recommendations:

### Recall@k:
  - Definition: Recall@k evaluates the fraction of relevant items retrieved among the top k recommendations over total relevant items. It measures the system's ability to find all relevant items in a fixed-sized list.
  - Use Case: In information retrieval and recommendation systems, Recall@k is crucial when it's essential to ensure that no relevant items are missed in the top k recommendations.

### Precision@k:

  - Definition: Precision@k assesses the fraction of retrieved items that are relevant among the top k recommendations. It measures the system's ability to provide relevant content at the top of the list.
  - Use Case: Precision@k is vital when there's a need to present users with highly relevant content in the initial recommendations. It helps in reducing user frustration caused by irrelevant suggestions.

### Mean Reciprocal Rank (MRR):

  - Definition: MRR measures the effectiveness of a system in ranking the most relevant items at the top of a list. It calculates the average of reciprocal ranks of the first correct item found in each ranked list of results: 
  MRR = 1/m \Sum(1/rank_i)
  - Use Case: MRR is often used in search and recommendation systems to assess how quickly users find relevant content. It's particularly useful when there is only one correct answer or when the order of results matters.

### Mean Average Precision (mAP):

  - Definition: mAP computes the average precision across multiple queries or users. Precision is calculated for each query, and the mean of these precisions is taken to provide a single performance score.
  - Use Case: mAP is valuable in scenarios where there are multiple users or queries, and you want to assess the overall quality of recommendations or search results across a diverse set of queries. mAP works well for binary relevances. For continues scores, we use nDCG. 

### Discounted Cumulative Gain (DCG):
  - Definition: Discounted Cumulative Gain (DCG) is a widely used evaluation metric primarily applied in the fields of information retrieval, search engines, and recommendation systems.
    - DCG quantifies the quality of a ranked list of items or search results by considering two key aspects:
      1. Relevance: Each item in the list is associated with a relevance score, which indicates how relevant it is to the user's query or preferences. Relevance scores are typically on a scale, with higher values indicating greater relevance.
      2. Position: DCG takes into account the position of each item in the ranked list. Items appearing higher in the list are considered more important because users are more likely to interact with or click on items at the top of the list.
    - DCG calculates the cumulative gain by summing the relevance scores of items in the ranked list up to a specified position.
    - To reflect the decreasing importance of items further down the list, DCG applies a discount factor, often logarithmic in nature.
  - Use case: 
    - DCG is employed to evaluate how effectively a system ranks and presents relevant items to users.
    - It is instrumental in optimizing search and recommendation algorithms, ensuring that highly relevant items are positioned at the top of the list for user engagement and satisfaction.

### Normalized Discounted Cumulative Gain (nDCG):

  - Definition: nDCG measures the quality of a ranked list by considering the graded relevance of items. It discounts the relevance of items as they appear further down the list and normalizes the score. It is calculated as the fraction of DCG over the Ideal DCG(IDCG) for an ideal ranking. 
  - Use Case: nDCG is beneficial when relevance is not binary (i.e., there are degrees of relevance), and you want to account for the diminishing importance of items lower in the ranking.

# Cross Entropy and Normalized Cross Entropy 
- The CE (also a loss function), measures how well the predicted probabilities align with the true class labels. It's defined as:

    - For binary classification:
    CE = - [y * log(p) + (1 - y) * log(1 - p)]
    
    - For multi-class classification:
    CE = - Σ(y_i * log(p_i))
    
    Where:
    - y is the true class label (0 or 1 for binary, one-hot encoded vector for multi-class).
    - p is the predicted probability assigned to the true class label.
    - The negative sign ensures that the loss is minimized when the predicted probabilities match the true labels. (the lower the better)
- NCE: CE(ML model) / CE(simple baseline)

### Ranking:
* Precision @k and Recall @k not a good fit (not consider ranking quality of out) 
* MRR, mAP, and nDCG good: 
  * MRR: focus on rank of 1st relevant item 
  * nDCG: relevance b/w user and item is non-binary 
  * mAP: relevance is binary 
* Ads ranking: NCE 
  
# Online metrics 
* CTR 


- Definition:

    - Click-Through Rate (CTR) is a metric that quantifies user engagement with a specific item or element, such as an advertisement, a search result, a recommended product, or a link.
    - It is calculated by dividing the number of clicks on the item by the total number of impressions (or views) it received.
    - Formula for CTR:
      CTR= Number of Clicks/Number of Impressions ×100%

    - Impressions: Impressions refer to the total number of times the item was displayed or viewed by users. For ads, it's the number of times the ad was shown to users. For recommendations, it's the number of times an item was recommended to users.

- Use Cases:
  - Online Advertising campaigns: widely used to assess how well ads are performing. A high CTR indicates that the ad is compelling and relevant to the target audience.
  - Recommendation Systems: CTR is used to measure how effectively recommended items attract user clicks.
- Search Engines: CTR is used to evaluate the quality of search results. High CTR for a search result indicates that it was relevant to the user's query.

* Conversion Rate: Conversion Rate measures the percentage of users who take a specific desired action after interacting with an item, such as making a purchase, signing up for a newsletter, or filling out a form. It helps assess the effectiveness of a call to action.

* Bounce Rate: Bounce Rate calculates the percentage of users who visit a webpage or view an item but leave without taking any further action, such as navigating to another page or interacting with additional content. A high bounce rate may indicate that users are not finding the content engaging.

* Engagement Rate: Engagement Rate evaluates the level of user interaction and participation with content or ads. It can include metrics like comments, shares, likes, or time spent on a webpage. A high engagement rate suggests that users are actively involved with the content.

* Time on Page: Time on Page measures how long users spend on a webpage or interacting with a specific piece of content. It helps evaluate user engagement and the effectiveness of content in holding user attention.

* Return on Investment (ROI): ROI assesses the financial performance of an advertising or marketing campaign by comparing the costs of the campaign to the revenue generated from it. It's crucial for measuring the profitability of marketing efforts.