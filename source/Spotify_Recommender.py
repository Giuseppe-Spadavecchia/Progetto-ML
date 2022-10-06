import numpy as np
import pandas as pd
from plotly import graph_objects as go
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

#Il file contiene i metodi necessari per effettuare le raccomandazioni, stamparle a video e per il calcolo delle
# metriche di valutazione

def make_recommendations(data, data_new, index, model, number_recommendation=5):

    #Si considerano gli elemti in base all'indice dato in input
    query = data_new.loc[index].to_numpy().reshape(1, -1)
    #Calcolo delle distanze e degli indici che puntano agli elementi legati a queste distanze
    distances, indices = model.kneighbors(query, n_neighbors=number_recommendation)
    distances = distances.reshape(number_recommendation, 1)
    #Si rimuove il primo elemento in quanto si rifericìsce al brano di cui si volgiono le raccomandazioni
    distances = np.delete(distances, 0)

    for i in indices:
        #Inserimento del nome del brano e dell'artista nella variabile delle raccomandazioni
        recommendations = data[['name', 'artists']].loc[i].where(data['id'] != index).dropna()
        recommendations['distances'] = distances

    return recommendations

def model_fit(df):

    model_knn = NearestNeighbors(algorithm='brute', n_neighbors=10)
    song_matrix = csr_matrix(df.values)
    model_knn.fit(song_matrix)

    return model_knn

def print_recommendations(dataframe, query_index, recommendations):

    #Scelta della canzone usando l'indice
    sample = dataframe.loc[dataframe['id'] == query_index]

    song = sample.iloc[0]['name']
    artists = sample.iloc[0]['artists']

    print('La traccia musicale considerata è', song, 'eseguita da', artists)

    print(recommendations[['name', 'artists']])

def evaluation_metrics(recommendations, test_elements, number_recommendation):
    number_relevant_elements = []
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

    # Stampa dei valori di recall e precision
    print('Il valore di precision calcolato è:', precision_rec_song)
    print('Il valore di recall calcolato è:', recall_rec_song)

    return precision_rec_song, recall_rec_song

#Stampa tramite grafico dei valri di precision e recall considerando due insiemi di artisti e canzoni
def print_metric(precision_recall_X, precision_Y, recall_Y):

    figure = go.Figure(
        data=[
            go.Bar(name="Precision",
                   x=precision_recall_X,
                   y=precision_Y),
            go.Bar(name="Recall",
                   x=precision_recall_X,
                   y=recall_Y)
        ])
    figure.update_layout(barmode='group', title='Metriche di valutazione',
                         yaxis_title='Precision e Recall in percentuale',
                         xaxis_title="Tipo di Raccomandazioni")
    figure.show()

