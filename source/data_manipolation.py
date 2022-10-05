import numpy as np
import pandas as pd

#Il file contiene i metodi con cui modificare il dataframe iniziale per la fase di training del modello

def manipolation(dataframe):

    # Si riducono le feature del dataframe a quelle considerate rilevanti
    df = dataframe.drop(
            ['name', 'artists', 'genre', 'popularity', 'duration_ms', 'time_signature', 'key', 'mode'], axis=1)

    # Normalizzazione delle feature tramite Z-Score
    df['loudness'] = (df['loudness'] - df['loudness'].mean()) / df['loudness'].std()
    df['tempo'] = (df['tempo'] - df['tempo'].mean()) / df['tempo'].std()
    df.index = df['id']
    df = df.drop(['id'], axis=1)

    return df



