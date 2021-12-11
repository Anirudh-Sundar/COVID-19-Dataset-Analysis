# COVID-19-Dataset-Analysis

### :busts_in_silhouette: Description
* Objective of the research involves Data anlalysis of articles in the CORD-19 Dataset (https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge). From the analysis we intend to gather research articles based on their subject (which is represented by the abstract).
* This analysis is performed by clustering the sentence embeddings and optimizing them by using a Genetic Algorithm.

### :cd: Repository Structure

* The *Metrics* folder contains the Silhouette Coefficient and the DBindex for the k-means clustering.
* The *Scripts* folder contains the scripts to convert the abstracts into embeddings.
* The *Dataset* folder contains the abstracts represented by 14 different types of embeddings.
* Run parallel_GA.ipynb and Kmeans_Using_Library.ipynb execute the Genetic algorithm and the Clustering of the data. 

### :books: Libraries Used
* Numpy
* pyspark
* nltk
* sklearn
* huggingface Transformers

### :computer: Maintained and Created by:

* Anirudh S (https://github.com/Anirudh-Sundar), BITS Pilani Hyd Campus
* Karthik Suresh (https://github.com/karths8), BITS Pilani Hyd Campus



