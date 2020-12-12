# cmpe493-term-project
cmpe493 fall 2020 term project : free text based ranked retrieval model 

Drive : https://drive.google.com/drive/folders/1iHdwsrQdw_25uPSN5zqcxAgIv-maFBD4?usp=sharing

 ## raw_data.csv

![image](https://raw.githubusercontent.com/egirgin/cmpe493-term-project/main/raw_data_sample.jpg?token=AGARMYEFCO5JKEPSAOEVDGS73NHQY)


# Ideas:

## Preprocessing:
 - Tokenization
 - Sentence Splitting (?)
 - Stemming
 - Lemmatization (?)
 - Normalization (punctuation removal, case folding, etc.)
 - Stopword Elimination

## Models:
 - Jaccard Coefficient: (Lec6)
    - ``` jaccard(query.tokens, document.tokens) ``` returns a score between 0 and 1
    - does not care multiple occurrences
---
 - Bag of Words: (Lec6)
    - counts the frequency of a token in a text
    - does not care ordering
    - term frequency, ```tf(t,d)``` : the number of times that term t occurs in document d
---
- Log freq. weighting: (Lec6)
    - ```w(t,d) = 1 + log10(tf(t,d)) if tf(t,d) > 0, else 0```
    - score every document-query pair: ```score(q, d) = sum( for all term t in (query q INTERSECTION document d), w(t, d) )```
---
- idf weighting: (Lec6)
    - document freq. ```df(t)``` : the number of documents that contain t
    - df is inverse measure of the informativeness of t
    - inverse document frequency ```idf(t) = log10(N/df(t))``` where N is total # of docs
    - measures the informativeness of t
---
- tf-idf weighting: (Lec6)
    - ```tf_idf(t,d) = w(t,d) * idf(t)```
    - ```score(q,d) = sum( for all term t in (query q INTERSECTION document d), tf_idf(t, d))```
    - see ```sklearn.feature_extraction.text.TfidfVectorizer```
---
- Vector representation: (Lec6)
    - Table : Rows are tokens, columns are documents -> calculate tf_idf(t,d) for each cell
    - Rows (terms) are axes of the space, documents are the vectors
    - Do the same to the queries
    - Euclidean distance is bad idea bcs it's sensitive to length of vector
    - Cosine Similiarity:
        - Length Normalization: ie. L2 norm (see ```sklearn.preprocessing.normalize```)
        - Apply cosine similarity: ```cosine_similarity(q,d)``` where q and d are vectors explained above (see ```sklearn.metrics.pairwise.cosine_similarity```)
        - After normalization, cosine similarity is just the **dot product**
        - Score for each document can divided into document's length (maybe softmax ?)
    - Practical considerations:
        - Consider ```w(t, q) = 1``` for queries : Faster Cosine
        - Take only high-idf query terms : documents came from low-idf terms are eleminated
        - Take a doc if cardinality of intersection between terms in query and terms in doc is higher than a threshold, say 3 or 4
        - champions list (top docs) : apply a threshold to the posting list of a term
        - we assume authority (quality) of each document is the same. thus we dont need a tier 

---
 - Deep Learning:
    - word embeddings
    - Word2Vec
    - BERT
    - LSTM/GRU/Attention

## Evaluation
 - Mean Average Precision (mAP)
 - Normalized Discounted Cumulative Gain (NDCG)
 - Precision of top 10 results (P@10)




