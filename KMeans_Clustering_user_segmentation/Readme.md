Core Fans Identification Using KMeans Clustering

This project focuses on identifying "core fans" from a streaming platform (e.g., YouTube, Netflix, or Patreon) by applying KMeans clustering to user engagement data. 
The goal is to segment users into distinct clusters, including current and potential future core fans, as well as at-risk fans, based on features such as:
- Duration watched,
- Ratings,
- Playback quality,
- Subscription type,
- and interactions.

The analysis is carried by following steps:
- Step 1: Install and import the needed packages
- Step 2: Download the dataset from Kaggle
- Step 3: Data pre-processing
  - 3(a). Load the data
  - 3(b). Inspect the data for missing values, duplication
  - 3(c) Check if each 'session_ID' is unique to 'user_ID'
- Step 4 Visualize the data
- Step 5: Feature Engineering
  - 5(a) Select users with 'Playback Quality' = 4K
  - 5(b) Select users who have provided Ratings = 5
  - 5(c) Split the data into two categories based on 'Subscription_Status'
  - 5(d) Select the features for clustering
  - 5(e) Apply standard scaling
- Step 6: KMeans Clustering
  - 6(a) Find optimal number of clusters based on WCSS (Elbow Method)
  - 6(b) Apply KMeans
  - 6(c) Visualize the clustering
  - 6(d) Get number of data per cluster
- Step 7: Conclusion
- Step 8: Future Work
  - Step 8(a): Analyze current ‘core fans’ based on following:
  - Step 8(b): Analyze future ‘core fans’:
  - Step 8(c) Re-analysis of this project
