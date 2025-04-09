import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by='GAME_DATE')
    return df
