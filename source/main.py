import numpy as np
import pandas as pd
from data_manipolation import manipolation
from Spotify_Recommender import *

np.random.seed(10)

#Lettura dei dati
dataframe = pd.read_csv('~/Documents/Progetto-ML/data_cleaned/data_wgenre_cleaned.csv')
dataframe.rename(columns={'artist_name':'artists', 'track_name':'name', 'track_id':'id'}, inplace=True)


#Manipolazione dei dati
df = manipolation(dataframe)

#Applicazione dell' algortimo KNN
model_knn = model_fit(df)

number_of_recommendation = 10

#Creazione delle raccomandazioni per quattro tipi di utenti
movie = dataframe[['id', 'name', 'artists']].where(dataframe['genre'] == 'Movie').dropna()
movie_index = np.random.choice(movie['id'])
movie_listener_rec = make_recommendations(dataframe, df, movie_index, model_knn, number_of_recommendation)


jazz = dataframe[['id', 'name', 'artists']].where(dataframe['genre'] == 'Jazz').dropna()
jazz_index = np.random.choice(jazz['id'])
jazz_listener_rec = make_recommendations(dataframe, df, jazz_index, model_knn, number_of_recommendation)


anime = dataframe[['id', 'name', 'artists']].where(dataframe['genre'] == 'Anime').dropna()
anime_index = np.random.choice(anime['id'])
anime_listener_rec = make_recommendations(dataframe, df, anime_index, model_knn, number_of_recommendation)


#Stampa delle raccomandazioni per singolo utente

print('Raccomandazioni per utente "movie":')
print_recommendations(dataframe, movie_index, movie_listener_rec)

print('Raccomandazioni per utente "jazz":')
print_recommendations(dataframe, jazz_index, jazz_listener_rec)

print('Raccomandazioni per utente "anime":')
print_recommendations(dataframe, anime_index, anime_listener_rec)

#Calcolo delle metriche di valutazione con la creazione di due utenti fittizzi

anime_listener_song = [
    'STARTRAiN!', 'Ichigo Parfait Ga Tomaranai', 'Idola No Circus', 'Monster Hunter',
    'BREAK IT!', 'tomoshibi', 'Around the world', 'The Seventh Gate' 'Wake up!',
    'Step Ahead', 'Brave Shine', 'One Winged Angel (From "Final Fantasy VII")', 'Guren no Yumiya']
anime_listener_artists = ["Mamoru Miyano", "Linked Horizon", "Capcom Sound Team", "AAA",
                          "LiSA", "IKASAN", "Aimer", "Kobasolo", "Falcom Sound Team jdk", "amazarashi", "buzzG"]

print('Metriche di valutazione per le canzoni utente "anime":')
precision_anime_song, recall_anime_song = evaluation_metrics(anime_listener_rec['name'], anime_listener_song, number_of_recommendation)
print('Metriche di valutazione per gli artisti:')
precision_anime_artists, recall_anime_artists = evaluation_metrics(anime_listener_rec['artists'], anime_listener_artists, number_of_recommendation)

movie_listener_song = [
    "Cradle and All ", "C'est beau de faire un Show", "Ma Doudou ", "Mangala Aarti",
    "Ultra Man 80", "Let Me Let Go", "Clic clac cloc", "Papa loves mambo", "Chanson pour les enfants l'hiver",
    "Wanna Be", "Forgotten Dreams"]
movie_listener_artists = ["Audra McDonald", "Leopold Stokowski", "Henri Salvador", "Bernard Minet",
                          "Chorus", "Richard M. Sherman", "Wanna Be", "Randy Newman",
                          "Cliff Edwards", "Dominique Tirmont", "Karine Costa"]

print('Metriche di valutazione per le canzoni utente "movie":')
precision_movie_song, recall_movie_song = evaluation_metrics(movie_listener_rec['name'], movie_listener_song, number_of_recommendation)
print('Metriche di valutazione per gli artisti:')
precision_movie_artists, recall_movie_artists = evaluation_metrics(movie_listener_rec['artists'], movie_listener_artists, number_of_recommendation)


#Stampa del grafico sulle metriche di valutazione
precision_recall_X = ['Movie artists', 'Anime artists', 'Movie song', 'Anime song']
precision_Y = [precision_movie_artists, precision_anime_artists, precision_movie_song, precision_anime_song]
recall_Y = [recall_movie_artists, recall_anime_artists, recall_movie_song, recall_anime_song]

print_metric(precision_recall_X, precision_Y, recall_Y)
