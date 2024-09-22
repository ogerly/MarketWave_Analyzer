import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path):
    columns = ['Datum', 'Zuletzt', 'Eröffn.', 'Hoch', 'Tief', 'Vol.', '+/- %']
    df = pd.read_csv(file_path, skiprows=1, names=columns)
    df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y')
    df['Zuletzt'] = df['Zuletzt'].str.replace('.', '').str.replace(',', '.').astype(float)
    df = df.iloc[::-1].reset_index(drop=True)
    df = df.set_index('Datum')
    df = df.asfreq('D')  # Setzt die Frequenz auf täglich
    df.rename(columns={'Zuletzt': 'Preis'}, inplace=True)
    return df

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    return result[1] <= 0.05  # p-value <= 0.05 indicates stationarity

def find_optimal_arima_order(timeseries, max_p=5, max_d=2, max_q=5):
    best_aic = float('inf')
    best_order = None
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(timeseries, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                except:
                    continue
    return best_order

def arima_forecast(data, forecast_days=5):
    if not check_stationarity(data):
        order = find_optimal_arima_order(data)
    else:
        order = (1, 0, 1)
    
    model = ARIMA(data, order=order)
    results = model.fit()
    forecast = results.forecast(steps=forecast_days)
    return forecast

def enhanced_trading_decision(forecast, ma_signal):
    arima_trend = 'Aufwärts' if forecast.iloc[-1] > forecast.iloc[0] else 'Abwärts'
    
    if ma_signal == 'Kauf' and arima_trend == 'Aufwärts':
        return 'Starkes Kaufsignal'
    elif ma_signal == 'Verkauf' and arima_trend == 'Abwärts':
        return 'Starkes Verkaufsignal'
    elif ma_signal != arima_trend:
        return 'Gemischte Signale - Vorsicht geboten'
    else:
        return 'Schwaches ' + ma_signal + 'signal'
    

def analyze_bitcoin_data(df, start_year, end_year):
    results = []
    for year in range(start_year, end_year + 1):
        train = df[df.index.year < year]
        test = df[df.index.year == year]
        
        if len(train) > 0 and len(test) > 0:
            forecast = arima_forecast(train['Preis'], len(test))
            
            # Simuliere MA-Signale (Dies sollte durch Ihre tatsächliche MA-Logik ersetzt werden)
            ma_signals = np.random.choice(['Kauf', 'Verkauf'], size=len(test))
            
            trading_decisions = [enhanced_trading_decision(forecast[:i+1], signal) 
                                 for i, signal in enumerate(ma_signals)]
            
            results.append({
                'year': year,
                'actual': test['Preis'],
                'forecast': forecast,
                'decisions': trading_decisions
            })
    
    return results



def arima_forecast(data, forecast_days=5, max_order=3):
    best_aic = float('inf')
    best_model = None
    
    for p in range(max_order + 1):
        for d in range(2):  # 0 oder 1 für Differenzierung
            for q in range(max_order + 1):
                try:
                    model = SARIMAX(data, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
                    results = model.fit(disp=False, maxiter=200)
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_model = results
                except:
                    continue
    
    if best_model is None:
        raise ValueError("Kein passendes Modell gefunden")
    
    forecast = best_model.forecast(steps=forecast_days)
    return forecast

# Hauptausführung
file_path = 'BTC__USD_daily.csv'
df = load_and_prepare_data(file_path)
start_year = df.index.year.min()
end_year = df.index.year.max()

analysis_results = analyze_bitcoin_data(df, start_year, end_year)

# Visualisierung
plt.figure(figsize=(15, 10))
for result in analysis_results:
    plt.plot(result['actual'].index, result['actual'], label=f'Actual {result["year"]}')
    plt.plot(result['forecast'].index, result['forecast'], label=f'Forecast {result["year"]}', linestyle='--')

plt.legend()
plt.title('ARIMA Forecasts vs Actual Prices with Trading Decisions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.yscale('log')
plt.grid(True)
plt.show()

# Ausgabe der Ergebnisse
for result in analysis_results:
    mae = abs(result['actual'] - result['forecast']).mean()
    print(f"Year {result['year']}:")
    print(f"  Mean Absolute Error: {mae:.2f}")
    print(f"  Trading Decisions: {result['decisions'][:5]}...")  # Zeigt die ersten 5 Entscheidungen
    print()