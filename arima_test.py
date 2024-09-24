import pandas as pd
import numpy as np
import requests
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------
# Anpassbare Parameter
# ---------------------------

# Liste der gleitenden Durchschnitte und Farben (für stündliche Daten)
ma_windows = [9, 20, 50, 100, 200, 400]
ma_colors = ['yellow', 'red', 'orange', 'green', 'purple', 'turquoise']
ma_labels = [f'MA {window}' for window in ma_windows]

# Schwellenwert für minimale Abstände zwischen den MAs bei der Bodenerkennung
distance_threshold = 0.005  # Kann angepasst werden

# ---------------------------
# Daten einlesen und vorbereiten
# ---------------------------

def get_hourly_data(symbol, interval, limit):
    base_url = 'https://api.binance.com'
    endpoint = '/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(base_url + endpoint, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    df['Datum'] = pd.to_datetime(df['Close Time'], unit='ms')
    df['Preis'] = df['Close'].astype(float)
    df = df[['Datum', 'Preis']]
    return df

# Beispielaufruf
symbol = 'BTCUSDT'
interval = '1h'  # 1-Stunden-Intervall
limit = 1000     # Anzahl der Datenpunkte (kann angepasst werden)

df = get_hourly_data(symbol, interval, limit)

# ---------------------------
# Berechnung der gleitenden Durchschnitte
# ---------------------------

for window, label in zip(ma_windows, ma_labels):
    df[label] = df['Preis'].rolling(window=window).mean()

# Abstände zwischen den MAs berechnen
df['Dist_MA9_MA20'] = df['MA 9'] - df['MA 20']
df['Dist_MA20_MA50'] = df['MA 20'] - df['MA 50']
df['Dist_MA50_MA100'] = df['MA 50'] - df['MA 100']
df['Dist_MA100_MA200'] = df['MA 100'] - df['MA 200']
df['Dist_MA200_MA400'] = df['MA 200'] - df['MA 400']

# ---------------------------
# Funktionen für ARIMA-Modell
# ---------------------------

from statsmodels.tsa.stattools import adfuller

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    return result[1] <= 0.05  # p-Wert <= 0.05 deutet auf Stationarität hin

def arima_forecast(timeseries, steps=5):
    # Überprüfen auf Stationarität
    if not check_stationarity(timeseries):
        d = 1  # Differenzierung
    else:
        d = 0
    # ARIMA-Modell anpassen
    try:
        model = ARIMA(timeseries, order=(1, d, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except:
        # Falls ein Fehler auftritt, leere Prognose zurückgeben
        return pd.Series()

# ---------------------------
# Funktionen zur Mustererkennung
# ---------------------------

# Funktion zur Erkennung von Kauf- und Verkaufsignalen mit ARIMA-Bestätigung
def detect_buy_sell_signals(df):
    signals = []
    for i in range(max(ma_windows), len(df)):
        # Original-Signale basierend auf MAs
        if df['MA 20'].iloc[i] > df['MA 50'].iloc[i] and df['MA 20'].iloc[i - 1] <= df['MA 50'].iloc[i - 1]:
            signal = 'Kauf'
        elif df['MA 20'].iloc[i] < df['MA 50'].iloc[i] and df['MA 20'].iloc[i - 1] >= df['MA 50'].iloc[i - 1]:
            signal = 'Verkauf'
        else:
            continue

        # ARIMA-Prognose abrufen
        recent_data = df['Preis'].iloc[:i]
        forecast = arima_forecast(recent_data)
        if forecast.empty:
            continue  # Wenn keine Prognose verfügbar ist, überspringen

        arima_trend = 'Aufwärts' if forecast.iloc[-1] > df['Preis'].iloc[i] else 'Abwärts'

        # Signal bestätigen
        if (signal == 'Kauf' and arima_trend == 'Aufwärts') or (signal == 'Verkauf' and arima_trend == 'Abwärts'):
            signals.append((df['Datum'].iloc[i], signal))
        else:
            # Signal ignorieren oder als schwach markieren
            pass
    return signals

# ---------------------------
# Investitionsbetrag erfassen
# ---------------------------

# Eingabe des Investitionsbetrags
investitionsbetrag = float(input("Bitte geben Sie den Investitionsbetrag in USD ein: "))

# ---------------------------
# Mustererkennung durchführen
# ---------------------------

signals = detect_buy_sell_signals(df)

# ---------------------------
# Gewinnberechnung
# ---------------------------

def calculate_profit(df, signals, investitionsbetrag):
    balance = investitionsbetrag
    btc_holding = 0
    last_action = None
    trade_history = []

    for date, signal in signals:
        preis = df.loc[df['Datum'] == date, 'Preis'].values[0]
        if signal == 'Kauf' and last_action != 'Kauf':
            # Kaufen
            btc_holding = balance / preis
            balance = 0
            last_action = 'Kauf'
            trade_history.append({'Datum': date, 'Aktion': 'Kauf', 'Preis': preis, 'BTC': btc_holding, 'Balance': balance})
        elif signal == 'Verkauf' and last_action == 'Kauf':
            # Verkaufen
            balance = btc_holding * preis
            btc_holding = 0
            last_action = 'Verkauf'
            trade_history.append({'Datum': date, 'Aktion': 'Verkauf', 'Preis': preis, 'BTC': btc_holding, 'Balance': balance})

    # Am Ende alles verkaufen, falls noch BTC gehalten werden
    if btc_holding > 0:
        preis = df['Preis'].iloc[-1]
        balance = btc_holding * preis
        trade_history.append({'Datum': df['Datum'].iloc[-1], 'Aktion': 'Verkauf (Ende)', 'Preis': preis, 'BTC': 0, 'Balance': balance})
        btc_holding = 0

    gesamtgewinn = balance - investitionsbetrag
    return gesamtgewinn, trade_history

gesamtgewinn, trade_history = calculate_profit(df, signals, investitionsbetrag)

print(f"\nGesamtgewinn: {gesamtgewinn:.2f} USD")

# ---------------------------
# Visualisierung
# ---------------------------

# Subplots erstellen: 2 Reihen (Hauptdiagramm und Flächendiagramm)
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        'BTC Kurs mit gleitenden Durchschnitten und Signalen',
        'Flächendiagramm der Abstände zwischen den MAs'
    )
)

# Den tatsächlichen BTC-Kurs als weiße Linie hinzufügen (erstes Diagramm)
fig.add_trace(
    go.Scatter(
        x=df['Datum'],
        y=df['Preis'],
        mode='lines',
        name='BTC Kurs',
        line=dict(color='white', width=2),
        hoverinfo='x+y',
    ),
    row=1, col=1
)

# Gesamt-MAs hinzufügen (sichtbar)
for label, color in zip(ma_labels, ma_colors):
    fig.add_trace(
        go.Scatter(
            x=df['Datum'],
            y=df[label],
            mode='lines',
            name=label,
            line=dict(color=color, width=2),
            hoverinfo='x+y',
            visible=True,
        ),
        row=1, col=1
    )

# Kauf- und Verkaufsignale markieren
for date, signal in signals:
    preis = df.loc[df['Datum'] == date, 'Preis'].values[0]
    color = 'green' if signal == 'Kauf' else 'red'
    symbol = 'arrow-up' if signal == 'Kauf' else 'arrow-down'
    fig.add_trace(
        go.Scatter(
            x=[date],
            y=[preis],
            mode='markers',
            marker=dict(symbol=symbol, color=color, size=12),
            name=signal,
            showlegend=False,
            hovertemplate=f'{signal}: {date.strftime("%d.%m.%Y %H:%M")}'
        ),
        row=1, col=1
    )

# ARIMA-Prognosen hinzufügen
forecast_steps = 5  # Anzahl der Stunden in die Zukunft
forecast_dates = pd.date_range(df['Datum'].iloc[-1], periods=forecast_steps+1, freq='H')[1:]
forecast_values = arima_forecast(df['Preis'], steps=forecast_steps)

if not forecast_values.empty:
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='ARIMA Prognose',
            line=dict(color='cyan', dash='dash'),
        ),
        row=1, col=1
    )

# Gestapeltes Flächendiagramm der Abstände zwischen den MAs hinzufügen (zweites Diagramm)
abstand_traces = [
    ('Dist_MA9_MA20', 'yellow', 'Abstand MA 9-20'),
    ('Dist_MA20_MA50', 'red', 'Abstand MA 20-50'),
    ('Dist_MA50_MA100', 'orange', 'Abstand MA 50-100'),
    ('Dist_MA100_MA200', 'green', 'Abstand MA 100-200'),
    ('Dist_MA200_MA400', 'purple', 'Abstand MA 200-400')
]

for dist_label, color, name in abstand_traces:
    fig.add_trace(
        go.Scatter(
            x=df['Datum'],
            y=df[dist_label],
            mode='lines',
            name=name,
            line=dict(color=color, width=0),
            fill='tozeroy',
            stackgroup='one',
            hoverinfo='x+y',
            visible='legendonly',  # Standardmäßig ausgeblendet
        ),
        row=2, col=1
    )

# Layout anpassen
fig.update_layout(
    title='BTC Kurs und Abstände zwischen den gleitenden Durchschnitten (MAs)',
    xaxis=dict(
        rangeslider=dict(visible=True),
        type='date'
    ),
    yaxis=dict(
        title='Preis in USD'
    ),
    yaxis2=dict(
        title='Abstand'
    ),
    hovermode='x unified',
    legend=dict(
        title='Legende',
        orientation='v',
        x=1.02,
        y=1,
        bordercolor='white',
        borderwidth=1,
    ),
    template='plotly_dark',
    autosize=True,
)

# Gewinnanzeige hinzufügen
fig.add_annotation(
    xref='paper', yref='paper',
    x=0.5, y=-0.2,
    text=f"Investitionsbetrag: {investitionsbetrag:.2f} USD<br>Gesamtgewinn: {gesamtgewinn:.2f} USD",
    showarrow=False,
    font=dict(size=14, color='white'),
    align='center'
)

# Responsives Design aktivieren
config = {'responsive': True}

# Diagramm anzeigen
fig.show(config=config)
