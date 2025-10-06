import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# ------------------ Streamlit Page Config ------------------
st.set_page_config(page_title="THE PREDICTION App", layout="wide")
st.title("📊  Time Series Prediction App")

# ------------------ File Upload ------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validate dataset
    if 'YEAR' not in df.columns or 'ANNUAL' not in df.columns:
        st.error("The uploaded file must contain 'YEAR' and 'ANNUAL' columns.")
        st.stop()

    # Cleaning
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    for col in df.columns:
        if col != "YEAR":
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].mean(), inplace=True)

    df.dropna(subset=['YEAR'], inplace=True)
    df = df.sort_values('YEAR').reset_index(drop=True)

    st.success("Data loaded successfully!")
    st.dataframe(df.head())

    # ------------------ Sidebar ------------------
    st.sidebar.header("Options")

    # Select target column (ANNUAL or quarters)
    target_col = st.sidebar.selectbox(
        "Select column for forecasting",
        [col for col in df.columns if col != "YEAR"]
    )

    method = st.sidebar.selectbox(
        "Choose a Method",
        [
            "1. Least Squares Method",
            "2. Freehand Curve",
            "3. Method of Semi-Averages",
            "4. Fitting a Curve (Polynomial)",
            "5. Method of Moving Averages",
            "6. Ratio to Trend",
            "7. Ratio to Moving Trend",
            "8. ARIMA",
            "9. Linear Regression (ML)",
            "10. Random Forest",
            "11. LSTM"
        ]
    )

    window_size = None
    if method in ["5. Method of Moving Averages", "7. Ratio to Moving Trend"]:
        window_size = st.sidebar.number_input("Moving Average Window", min_value=2, value=3)

    predict_year = st.sidebar.number_input(
        "Enter the year to predict:",
        min_value=int(df['YEAR'].max()) + 1,
        value=int(df['YEAR'].max()) + 1
    )

    btn_apply = st.sidebar.button("Run Prediction", use_container_width=True)

# ------------------ Metrics Function ------------------
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2

def display_metrics(y_true, y_pred, method_name):
    mae, rmse, mape, r2 = calculate_metrics(y_true, y_pred)
    st.write(f"**{method_name} Metrics on Historical Data:**")
    st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, R²: {r2:.2f}")

# ------------------ Prediction Methods ------------------
def least_squares(df, predict_year, target_col):
    x = np.arange(len(df)).reshape(-1,1)
    y = df[target_col].values
    model = LinearRegression().fit(x, y)
    trend = model.predict(x)
    df['Trend'] = trend

    # Prediction
    steps = predict_year - int(df['YEAR'].max())
    year_index = len(df) + steps
    prediction = model.predict([[year_index]])[0]

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(df['YEAR'], df[target_col], label="Actual", color='blue')
    plt.plot(df['YEAR'], df['Trend'], label="Trend", color='red')
    plt.scatter(predict_year, prediction, color='green', s=100, label=f"Prediction {predict_year}")
    plt.xlabel("Year"); plt.ylabel(target_col); plt.title(f"Least Squares - {target_col}"); plt.legend()
    st.pyplot(plt.gcf())
    st.success(f"Predicted {target_col} for {predict_year}: {prediction:.2f}")

    # Metrics
    display_metrics(y, trend, "Least Squares")

def freehand_curve(df, predict_year, target_col):
    window = 5
    df['Smoothed'] = df[target_col].rolling(window=window, center=True).mean()
    smoothed = df['Smoothed'].dropna()
    prediction = smoothed.iloc[-1]

    plt.figure(figsize=(10,5))
    plt.plot(df['YEAR'], df[target_col], label="Actual", color='blue')
    plt.plot(df['YEAR'], df['Smoothed'], label="Smoothed", color='orange')
    plt.scatter(predict_year, prediction, color='green', s=100, label=f"Prediction {predict_year}")
    plt.xlabel("Year"); plt.ylabel(target_col); plt.title(f"Freehand Curve - {target_col}"); plt.legend()
    st.pyplot(plt.gcf())
    st.success(f"Predicted {target_col} for {predict_year}: {prediction:.2f}")

    # Metrics (drop NA for smoothed)
    display_metrics(df[target_col][2:-2], smoothed[2:-2], "Freehand Curve")

def semi_averages(df, predict_year, target_col):
    n = len(df)
    half = n // 2
    first_avg = df[target_col][:half].mean()
    second_avg = df[target_col][half:].mean()
    trend_line = [first_avg]*(half) + [second_avg]*(n-half)
    slope = second_avg - first_avg
    prediction = second_avg + slope

    plt.figure(figsize=(10,5))
    plt.plot(df['YEAR'], df[target_col], label="Actual", color='blue')
    plt.plot(df['YEAR'], trend_line, label="Semi-Average Trend", color='red')
    plt.scatter(predict_year, prediction, color='green', s=100, label=f"Prediction {predict_year}")
    plt.xlabel("Year"); plt.ylabel(target_col); plt.title(f"Semi-Averages - {target_col}"); plt.legend()
    st.pyplot(plt.gcf())
    st.success(f"Predicted {target_col} for {predict_year}: {prediction:.2f}")

    display_metrics(df[target_col], trend_line, "Semi-Averages")

def moving_average(df, window, predict_year, target_col):
    df['MA'] = df[target_col].rolling(window=window).mean()
    prediction = df['MA'].iloc[-1]

    plt.figure(figsize=(10,5))
    plt.plot(df['YEAR'], df[target_col], label="Actual", color='blue')
    plt.plot(df['YEAR'], df['MA'], label=f"MA({window})", color='purple')
    plt.scatter(predict_year, prediction, color='green', s=100, label=f"Prediction {predict_year}")
    plt.xlabel("Year"); plt.ylabel(target_col); plt.title(f"Moving Average - {target_col}"); plt.legend()
    st.pyplot(plt.gcf())
    st.success(f"Predicted {target_col} for {predict_year}: {prediction:.2f}")

    display_metrics(df[target_col][window-1:], df['MA'][window-1:], f"Moving Average ({window})")

# ------------------ ML/AI Forecasting ------------------
def arima_predict(df, predict_year, target_col):
    y_series = pd.Series(df[target_col].values, index=df['YEAR'])
    steps = predict_year - int(df['YEAR'].max()) + 5  # extend 5 years
    model = ARIMA(y_series, order=(1,1,1)).fit()
    forecast = model.forecast(steps=steps)
    pred_val = forecast.iloc[predict_year - int(df['YEAR'].max()) - 1]

    plt.figure(figsize=(12,6))
    plt.plot(df['YEAR'], df[target_col], label="Actual", color='blue')
    plt.plot(forecast.index, forecast.values, label="Forecast", color='red')
    plt.scatter(predict_year, pred_val, color='green', s=100, label=f"Prediction {predict_year}")
    plt.xlabel("Year"); plt.ylabel(target_col); plt.title(f"ARIMA Forecast - {target_col}"); plt.legend()
    st.pyplot(plt.gcf())
    st.success(f"ARIMA Prediction for {predict_year} ({target_col}): {pred_val:.2f}")

    # Metrics on historical data
    y_pred_hist = model.fittedvalues
    display_metrics(df[target_col], y_pred_hist, "ARIMA")

def lr_predict(df, predict_year, target_col):
    lr = LinearRegression().fit(df[['YEAR']], df[target_col])
    pred_val = lr.predict([[predict_year]])[0]

    future_years = np.arange(df['YEAR'].min(), predict_year + 6).reshape(-1,1)
    preds = lr.predict(future_years)

    plt.figure(figsize=(12,6))
    plt.plot(df['YEAR'], df[target_col], label="Actual", color='blue')
    plt.plot(future_years, preds, label="LR Trend", color='red')
    plt.scatter(predict_year, pred_val, color='green', s=100, label=f"Prediction {predict_year}")
    plt.xlabel("Year"); plt.ylabel(target_col); plt.title(f"Linear Regression - {target_col}"); plt.legend()
    st.pyplot(plt.gcf())
    st.success(f"Linear Regression Prediction for {predict_year} ({target_col}): {pred_val:.2f}")

    # Metrics
    y_pred_hist = lr.predict(df[['YEAR']])
    display_metrics(df[target_col], y_pred_hist, "Linear Regression")

def rf_predict(df, predict_year, target_col):
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(df[['YEAR']], df[target_col])
    pred_val = rf.predict([[predict_year]])[0]

    future_years = np.arange(df['YEAR'].min(), predict_year + 6).reshape(-1,1)
    preds = rf.predict(future_years)

    plt.figure(figsize=(12,6))
    plt.plot(df['YEAR'], df[target_col], label="Actual", color='blue')
    plt.plot(future_years, preds, label="RF Fit", color='red')
    plt.scatter(predict_year, pred_val, color='green', s=100, label=f"Prediction {predict_year}")
    plt.xlabel("Year"); plt.ylabel(target_col); plt.title(f"Random Forest - {target_col}"); plt.legend()
    st.pyplot(plt.gcf())
    st.success(f"Random Forest Prediction for {predict_year} ({target_col}): {pred_val:.2f}")

    # Metrics
    y_pred_hist = rf.predict(df[['YEAR']])
    display_metrics(df[target_col], y_pred_hist, "Random Forest")

def lstm_predict(df, predict_year, target_col):
    seq_len = 5
    data = df[target_col].values.reshape(-1,1)
    generator = TimeseriesGenerator(data, data, length=seq_len, batch_size=1)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_len,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=50, verbose=0)

    # Predict future
    steps = predict_year - int(df['YEAR'].max()) + 5
    last_seq = data[-seq_len:]
    preds = []
    for _ in range(steps):
        next_val = model.predict(last_seq.reshape(1, seq_len,1), verbose=0)[0][0]
        preds.append(next_val)
        last_seq = np.append(last_seq[1:], next_val)
    pred_val = preds[predict_year - int(df['YEAR'].max()) - 1]

    future_years = list(range(df['YEAR'].iloc[-1]+1, predict_year+6))
    plt.figure(figsize=(12,6))
    plt.plot(df['YEAR'], df[target_col], label="Actual", color='blue')
    plt.plot(future_years, preds, label="LSTM Forecast", color='red')
    plt.scatter(predict_year, pred_val, color='green', s=100, label=f"Prediction {predict_year}")
    plt.xlabel("Year"); plt.ylabel(target_col); plt.title(f"LSTM Forecast - {target_col}"); plt.legend()
    st.pyplot(plt.gcf())
    st.success(f"LSTM Prediction for {predict_year} ({target_col}): {pred_val:.2f}")

    # Historical metrics
    hist_preds = []
    for i in range(seq_len, len(data)):
        last_seq = data[i-seq_len:i]
        next_val = model.predict(last_seq.reshape(1, seq_len,1), verbose=0)[0][0]
        hist_preds.append(next_val)
    y_true_hist = data[seq_len:].flatten()
    display_metrics(y_true_hist, np.array(hist_preds), "LSTM")

# ------------------ Apply Method ------------------
if uploaded_file and btn_apply:
    if method == "1. Least Squares Method":
        least_squares(df, predict_year, target_col)
    elif method == "2. Freehand Curve":
        freehand_curve(df, predict_year, target_col)
    elif method == "3. Method of Semi-Averages":
        semi_averages(df, predict_year, target_col)
    elif method == "4. Fitting a Curve (Polynomial)":
        st.warning("Polynomial fitting not implemented yet")
    elif method == "5. Method of Moving Averages":
        moving_average(df, window_size, predict_year, target_col)
    elif method == "6. Ratio to Trend":
        st.warning("Ratio to Trend not implemented yet")
    elif method == "7. Ratio to Moving Trend":
        st.warning("Ratio to Moving Trend not implemented yet")
    elif method == "8. ARIMA":
        arima_predict(df, predict_year, target_col)
    elif method == "9. Linear Regression (ML)":
        lr_predict(df, predict_year, target_col)
    elif method == "10. Random Forest":
        rf_predict(df, predict_year, target_col)
    elif method == "11. LSTM":
        lstm_predict(df, predict_year, target_col)
