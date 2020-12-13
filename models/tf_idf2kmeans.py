import pandas as pd
import numpy as np
import json
import os

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def kmeans_relevance(similarity_matrix):
    """
        Using cosine similarities computes kmeans clustering

        :similarity_matrix: rows -> queries, columns -> document ids
        :returns: a dataframe -> rows: queries, columns -> documents ids, cells -> relevance level (0, 1, 2)
    """

    result_df = similarity_matrix

    k_means_results = []

    for index, row in similarity_matrix.iterrows():
        query_number = index*2+1
        query_similarities = row.to_numpy().reshape(-1,1)

        plt.plot(query_similarities)

        # Fit to Kmeans
        
        query_k_means = KMeans(n_clusters=3)

        query_k_means.fit_predict(query_similarities)

        # Process cluster centers

        query_centers = query_k_means.cluster_centers_.reshape(-1)

        query_centers = np.sort(query_centers)

        upper_bound = ( query_centers[1] +  query_centers[2] ) / 2

        lower_bound = ( query_centers[0] +  query_centers[1] ) / 2 

        plt.hlines(upper_bound, 0, len(query_similarities), colors="k", zorder=10)

        plt.hlines(lower_bound, 0, len(query_similarities), colors="k", zorder=10)
        
        # Determine relevances

        high_relevance = query_similarities >= upper_bound

        partially_relevance_upper = query_similarities < upper_bound

        partially_relevance_lower = query_similarities > lower_bound

        partially_relevance = partially_relevance_lower * partially_relevance_upper

        non_relevance = query_similarities <= lower_bound

        #print(high_relevance)

        high_relevance_docid = similarity_matrix.columns[high_relevance.reshape(-1)]

        partially_relevance_docid = similarity_matrix.columns[partially_relevance.reshape(-1)]

        non_relevance_docid = similarity_matrix.columns[non_relevance.reshape(-1)]

        plt.savefig("./models/kmeans/cos_sim_q{}.png".format(query_number))

        result_df.loc[index, high_relevance_docid] = 2

        result_df.loc[index, partially_relevance_docid] = 1

        result_df.loc[index, non_relevance_docid] = 0

        plt.clf()

    return result_df
        


def main():
    similarity_matrix = pd.read_csv("./preprocessing/cosine_similarity_matrix.csv", index_col=0)
    
    try:
        os.mkdir("./models/kmeans")
    except:
        pass
    kmeans_predictions = kmeans_relevance(similarity_matrix)

    kmeans_predictions.to_csv("./models/kmeans/kmeans_predictions.csv")


if __name__=="__main__": 
    main() 
