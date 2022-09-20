import numpy as np
import pandas as pd
from Spotify_Recommender import make_recommendations, print_recommendations, evaluation_metrics, model_fit
from data_manipolation import manipolation

#Seed usato per motivi di test dell'algoritmo
np.random.seed(42)

#Lettura dei dati
dataframe = pd.read_csv('~/Documents/Progetto_ML/data/data.csv')

#Manipolazione dei dati
df = manipolation(dataframe)

#Applicazione dell' algortimo KNN
model_knn = model_fit(df)

#Produzione e stampa a video delle raccomandazione
query_index = np.random.choice(df.index)
number_of_recommendation = 20

recommendations = make_recommendations(dataframe, df, query_index, model_knn, number_of_recommendation)
print_recommendations(dataframe, query_index, recommendations)


rec_song = recommendations['name']
rec_artists = recommendations['artists']

#Creazione dell'utente fitizzio con delle preferenze per quanto riguarda sia i brani che gli artsisti
test_artists = ["['Ignacio Corsini']", "['Beyoncé']", "['Georgius']", "['Louis Armstrong']", "['Kendrick Lamar']", "['Regina Belle']", "['Francisco Canaro']" ]
test_song = ["Summer Wit' Miami", "Good Life", "You're Driving Me Crazy", "Back O' Town Blues", "Sonata 1", "good kid", "Praise God I'm Satisfied", "Ring The Alarm"]

evaluation_metrics(rec_song, test_song, number_of_recommendation)
evaluation_metrics(rec_artists, test_artists, number_of_recommendation)











