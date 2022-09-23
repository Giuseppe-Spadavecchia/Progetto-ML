import numpy as np
import pandas as pd
from Spotify_Recommender import make_recommendations, print_recommendations, evaluation_metrics, model_fit
from data_manipolation import manipolation

#Seed usato per motivi di test dell'algoritmo
np.random.seed(42)

#Lettura dei dati
dataframe = pd.read_csv('~/Documents/Progetto-ML/data/data.csv')
dataframe = dataframe.drop_duplicates(subset=['name'])

dataframe.to_csv('~/Documents/Progetto-ML/data/dataframe.csv', index=False)
dataframe = pd.read_csv('~/Documents/Progetto-ML/data/dataframe.csv')

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
test_artists = ["['Rise Against']", "['Jason Aldean']", "['Matchbook Romance']", "['Louis Armstrong']", "['Eminem']", "['Regina Belle']", " ['LiSA']" ]
test_song = ["Life Less Frightening ", "Good Life", " Mt. Diablo ", "Back O' Town Blues", "crossing field ", "good kid", "Call To Arms ", "Crazy Town"]

evaluation_metrics(rec_song, test_song, number_of_recommendation)
evaluation_metrics(rec_artists, test_artists, number_of_recommendation)











