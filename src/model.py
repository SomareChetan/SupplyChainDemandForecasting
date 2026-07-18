import pandas as pd
from prophet import Prophet

def load_store_dept_data(csv_path, store_id, dept_id):
    """Load and filter data for a specific store and department."""
    df = pd.read_csv(csv_path)
    filtered = df[(df['Store'] == store_id) & (df['Dept'] == dept_id)].copy()
    filtered = filtered.sort_values('Date')
    return filtered

def prepare_prophet_data(store_df):
    """Convert to Prophet's required ds/y format."""
    prophet_df = store_df[['Date', 'Weekly_Sales']].rename(
        columns={'Date': 'ds', 'Weekly_Sales': 'y'}
    )
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    return prophet_df

def train_forecast_model(prophet_df, periods=12):
    """Train Prophet model and generate future forecast."""
    model = Prophet(yearly_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)
    return model, forecast

def evaluate_forecast(prophet_df, forecast):
    """Calculate MAE and RMSE for the historical period."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    actual = prophet_df['y'].values
    predicted = forecast['yhat'][:len(actual)].values
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse