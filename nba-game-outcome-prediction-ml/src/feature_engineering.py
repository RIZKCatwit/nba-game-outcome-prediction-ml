def add_features(df):
    df['PTS_DIFF_LAST_5'] = df['HOME_AVG_PTS_LAST_5'] - df['AWAY_AVG_PTS_LAST_5']
    df['WIN_STREAK_DIFF'] = df['HOME_WIN_STREAK'] - df['AWAY_WIN_STREAK']
    return df

def select_features(df):
    features = ['PTS_DIFF_LAST_5', 'WIN_STREAK_DIFF', 'HOME_REST_DAYS', 'AWAY_REST_DAYS']
    X = df[features].dropna()
    y = df.loc[X.index, 'HOME_TEAM_WIN']
    return X, y
