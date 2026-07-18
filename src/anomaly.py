from sklearn.ensemble import IsolationForest

def detect_anomalies(store_df, contamination=0.05):
    """Flag unusual weeks in sales data using Isolation Forest."""
    iso_model = IsolationForest(contamination=contamination, random_state=42)
    store_df = store_df.copy()
    store_df['anomaly'] = iso_model.fit_predict(store_df[['Weekly_Sales']])
    store_df['anomaly'] = store_df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    return store_df