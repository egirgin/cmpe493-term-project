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
    - word embeddings (https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca)
    - Word2Vec
    - BERT
    - LSTM/GRU/Attention
    - Glove

## Evaluation
 - Mean Average Precision (mAP)
 - Normalized Discounted Cumulative Gain (NDCG)
 - Precision of top 10 results (P@10)
 
### Installing trec_eval:
 1. Download trec_eval from https://trec.nist.gov/trec_eval/ .
 1. Extract the tar.gz file.
 1. In the trec eval directory, open a terminal and type "make".
 1. The name of the executable is trec_eval in the same directory. You can test it with:
  ```
    ./trec_eval test/qrels.test test/results.test
  ```
 
### Using trec_eval to evaluate:
 On terminal,
 ```
 <path-to-the-trec_eval> <path-to-qrel-file> <path-to-result-file>
 ```
 Example:
 ```
 ../trec_eval-9.0.7/trec_eval myqrels.txt myresults.txt
```

### Qrels and results file:

* First argument of the trec-eval (qrels file) should be the file that contains correct labels. It represents the ground truth. It has the format:

query-id 0 document-id relevance

example:

0 0 005b2j4b 2

0 0 00fmeepz 1

...

* Second argument of the trec-eval (results file) should be the file that contains our predictions. It has the format:

query-id	Q0	document-id	rank	score <explanation>

example:

0 Q0 2b73a28n 0 0 STANDARD

0 Q0 zjufx4fo 0 0 STANDARD

...

"Q0" and rank is currently ignored. Explanation is any sequence of alphanumeric characters that is used for identifying the run.

### Options:

-q: In addition to summary evaluation, give evaluation for each query

-l\<num\>: Num indicates the minimum relevance judgement value needed for a document to be called relevant. (All measures used by TREC eval are based on binary              relevance).  Used if trec_rel_file contains relevance judged on a multi-relevance scale.  Default is 1.

### Output

![measures](https://user-images.githubusercontent.com/33669453/102709347-ddf36c00-42ba-11eb-9b41-3a9ef41609f2.png)

# References

http://www.rafaelglater.com/en/post/learn-how-to-use-trec_eval-to-evaluate-your-information-retrieval-system
https://github.com/usnistgov/trec_eval
