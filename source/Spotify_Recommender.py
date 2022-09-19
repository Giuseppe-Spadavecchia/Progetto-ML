import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

#Il file contiene i metodi necessari per effettuare le raccomandazioni, stamparle a video e per il calcolo delle
# metriche di valutazione

def make_recommendations(data, data_new, index, model, number_recommendation=5):

    #loc serve a prendere un elemento per l'indice specificato
    query = data_new.loc[index].to_numpy().reshape(1, -1)
    #print('Ricerca di raccomandazioni:')
    distances, indices = model.kneighbors(query, n_neighbors=number_recommendation)
    distances = distances.reshape(number_recommendation, 1)
    #Si rimuove il primo elemento in quanto si rifericìsce al brano di cui si volgiono le raccomandazioni
    distances = np.delete(distances, 0)

    for i in indices:
        recommendations = data[['name', 'artists']].loc[i].where(data['id'] != index).dropna()
        recommendations['distances'] = distances

    return recommendations

def model_fit(df):

    model_knn = NearestNeighbors(algorithm='brute', n_neighbors=10)
    song_matrix = csr_matrix(df.values)
    model_knn.fit(song_matrix)

    return model_knn

def print_recommendations(dataframe, query_index, recommendations):

    #Scelta della canzone in maniera casuale usando l'indice
    sample = dataframe.loc[dataframe['id'] == query_index]

    song = sample.iloc[0]['name']
    artists = sample.iloc[0]['artists']

    print('La traccia musicale considerata è', song, 'eseguita da', artists)

    print(recommendations[['name', 'artists']])

def evaluation_metrics(recommendations, test_elements, number_recommendation):
    number_relevant_elements = []
    number_relevant_rec_artists = []
    n = len(test_elements)

    #Riempimento della variabile ccontenente valori binari per indicare quali elementi considerare rilevanti
    for value in recommendations:
        if value in test_elements:
            number_relevant_elements.append(1)
        else:
            number_relevant_elements.append(0)

    #Calcolo delle metriche di precision e di recall
    precision_rec_song = sum(number_relevant_elements) / number_recommendation
    recall_rec_song = sum(number_relevant_elements) / n

    print('Il valore di precision calcolato è:', precision_rec_song)
    print('Il valore di recall calcolato è:', recall_rec_song)




