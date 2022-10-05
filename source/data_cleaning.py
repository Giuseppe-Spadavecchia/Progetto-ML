import pandas as pd

#In questo file veine effettuata la creazione di nuovi file .csv che contengono i dati non duplicati


#Eliminazione dei duplicati dal dataset 'data_wgenre.csv'
dataframe = pd.read_csv('~/Documents/Progetto_ML/data/data_wgenre.csv')
dataframe = dataframe.drop_duplicates(subset=['track_id'])

dataframe.to_csv('~/Documents/Progetto_ML/data_cleaned/data_wgenre_cleaned.csv', index=False)