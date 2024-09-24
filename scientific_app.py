import pandas as pd
import numpy as np
import requests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------
# SETTINGS - Anpassbare Parameter
# ---------------------------

# 1. Datenquelle
USE_DAILY_DATA = True  # True für tägliche Daten, False für stündliche Daten

# 2. Investitionsbetrag
INVESTITIONSBETRAG = 1000.0  # Startkapital in USD

# 3. Liste der gleitenden Durchschnitte (MAs) und Farben
MA_WINDOWS = [9, 20, 50, 100, 200, 400]  # Perioden für MAs
MA_COLORS = ['yellow', 'red', 'orange', 'green', 'purple', 'turquoise']  # Farben für die Darstellung
MA_LABELS = [f'MA {window}' for window in MA_WINDOWS]

# 4. Kaufentscheidungs-Parameter
# Schwellenwerte für Preisrückgang seit letztem Verkauf (%)
MIN_PRICE_DROP_1 = -0.04  # Mindestens -4% Preisrückgang
MIN_PRICE_DROP_2 = -0.10  # Mindestens -10% Preisrückgang
MIN_PRICE_DROP_3 = -0.20  # Mehr als -20% Preisrückgang

# MA-Kreuzungsbedingungen
MA_CROSS_SHORT = (9, 20)  # (kurzer MA, langer MA)
MA_CROSS_MEDIUM = (50, 100)  # (kurzer MA, langer MA)
MA_CROSS_DAYS = 10  # Anzahl der Tage, innerhalb derer die Kreuzung erfolgen soll

# 5. Verkaufsentscheidungs-Parameter
# Mindestgewinn seit letztem Kauf (%)
MIN_PROFIT = 0.05  # Mindestens 5% Gewinn erforderlich für Verkauf

# MA-Kreuzungsbedingungen für Verkauf
SELL_MA_CROSS = (20, 50)  # (kurzer MA, langer MA)

# 6. Indikator-Schwellenwerte
D_THRESHOLD = 0.01  # Für MA_Distance in der Entscheidungsregel
C_THRESHOLD = 10    # Für Convergence_Indicator in der Entscheidungsregel

# 7. ARIMA-Modell Einstellungen
ARIMA_ORDER = (1, 1, 1)  # (p, d, q) Parameter für ARIMA-Modell

# ---------------------------
# ERKLÄRUNGEN DER PARAMETER
# ---------------------------

# USE_DAILY_DATA:
#   True  - Verwendet tägliche Daten (empfohlen für längere Zeiträume)
#   False - Verwendet stündliche Daten (detailliertere Analyse, aber mehr Datenpunkte)

# INVESTITIONSBETRAG:
#   Startkapital für die Simulation der Handelsstrategie.

# MA_WINDOWS:
#   Liste der Perioden für die gleitenden Durchschnitte, die in der Strategie verwendet werden.

# MA_COLORS:
#   Farben für die Darstellung der MAs im Diagramm.

# Kaufentscheidungs-Parameter:
# MIN_PRICE_DROP_1, MIN_PRICE_DROP_2, MIN_PRICE_DROP_3:
#   Schwellenwerte für den Preisrückgang seit dem letzten Verkauf, um Kaufentscheidungen zu treffen.

# MA_CROSS_SHORT, MA_CROSS_MEDIUM:
#   Paare von MAs, deren Kreuzung überwacht wird, um Kaufentscheidungen zu treffen.

# MA_CROSS_DAYS:
#   Anzahl der Tage, innerhalb derer bestimmte MA-Kreuzungen stattfinden sollen.

# Verkaufsentscheidungs-Parameter:
# MIN_PROFIT:
#   Mindestgewinn in Prozent, der seit dem letzten Kauf erzielt werden muss, um einen Verkauf in Betracht zu ziehen.

# SELL_MA_CROSS:
#   Paar von MAs, deren Kreuzung für Verkaufsentscheidungen überwacht wird.

# Indikator-Schwellenwerte:
# D_THRESHOLD, C_THRESHOLD:
#   Schwellenwerte für die Indikatoren MA_Distance und Convergence_Indicator in der Entscheidungsregel.

# ARIMA_ORDER:
#   (p, d, q)-Parameter für das ARIMA-Modell zur Prognose des Preistrends.

# ---------------------------
# DATEN EINLESEN UND VORBEREITEN
# ---------------------------

def get_daily_data():
    # Laden der historischen BTC/USD Daten aus CSV
    columns = ['Datum', 'Zuletzt', 'Eröffn.', 'Hoch', 'Tief', 'Vol.', '+/- %']
    df = pd.read_csv('BTC__USD_daily.csv', skiprows=1, names=columns)
    df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y')
    df['Zuletzt'] = df['Zuletzt'].str.replace('.', '').str.replace(',', '.').astype(float)
    df = df.iloc[::-1].reset_index(drop=True)
    df['Preis'] = df['Zuletzt']
    return df[['Datum', 'Preis']]

def get_hourly_data(symbol='BTCUSDT', interval='1h', limit=1000):
    # Abrufen der stündlichen Daten von Binance API
    base_url = 'https://api.binance.com'
    endpoint = '/api/v3/klines'
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
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

# Auswahl der Datenquelle basierend auf USE_DAILY_DATA
if USE_DAILY_DATA:
    df = get_daily_data()
else:
    df = get_hourly_data()

# Berechnung der gleitenden Durchschnitte
for window, label in zip(MA_WINDOWS, MA_LABELS):
    df[label] = df['Preis'].rolling(window=window).mean()

# Abstände zwischen den MAs berechnen
distance_columns = []
for i in range(len(MA_WINDOWS)):
    for j in range(i+1, len(MA_WINDOWS)):
        col_name = f'Dist_MA{MA_WINDOWS[i]}_MA{MA_WINDOWS[j]}'
        df[col_name] = df[MA_LABELS[i]] - df[MA_LABELS[j]]
        distance_columns.append(col_name)

# ---------------------------
# FUNKTIONEN FÜR INDIKATOREN UND ENTSCHEIDUNGSREGELN
# ---------------------------

def calculate_ma_distance(row, ma_labels):
    # Berechnet die durchschnittliche Distanz zwischen allen MAs
    distances = [abs(row[ma1] - row[ma2]) for i, ma1 in enumerate(ma_labels) for ma2 in ma_labels[i+1:]]
    return sum(distances) / len(distances)

def calculate_breakthrough_signal(df, ma_labels):
    # Berechnet das Durchbruchssignal basierend auf den MAs
    signals = pd.Series(0, index=df.index)
    for i in range(len(ma_labels)):
        for j in range(i+1, len(ma_labels)):
            signal = np.sign(df[ma_labels[i]] - df[ma_labels[j]]) - \
                     np.sign(df[ma_labels[i]].shift(1) - df[ma_labels[j]].shift(1))
            signals += signal.fillna(0)
    return signals

def calculate_convergence_indicator(df, ma_labels):
    # Berechnet den Konvergenzindikator der MAs
    ma_values = df[ma_labels]
    return 1 / ma_values.std(axis=1)

def check_stationarity(timeseries):
    # Überprüft die Stationarität einer Zeitreihe
    result = adfuller(timeseries)
    return result[1] <= 0.05  # p-Wert <= 0.05 deutet auf Stationarität hin

def arima_forecast(timeseries, steps=5):
    # ARIMA-Prognose der Zeitreihe
    if not check_stationarity(timeseries):
        d = 1  # Differenzierung
    else:
        d = 0
    try:
        model = ARIMA(timeseries, order=ARIMA_ORDER)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except:
        # Falls ein Fehler auftritt, leere Prognose zurückgeben
        return pd.Series()

def enhanced_decision_rule(row):
    # Entscheidungsregel basierend auf Indikatoren
    if row['MA_Distance'] < D_THRESHOLD and row['Convergence_Indicator'] > C_THRESHOLD:
        return 'Ausbruch erwartet'
    elif row['Breakthrough_Signal'] > 0 and row['Preis'] < row['MA 200']:
        return 'Kauf erwägen'
    elif row['Breakthrough_Signal'] < 0 and row['Preis'] > row['MA 200']:
        return 'Verkauf erwägen'
    else:
        return 'Neutral'

# ---------------------------
# INDIKATOREN BERECHNEN
# ---------------------------

df['MA_Distance'] = df.apply(lambda row: calculate_ma_distance(row, MA_LABELS), axis=1)
df['Breakthrough_Signal'] = calculate_breakthrough_signal(df, MA_LABELS)
df['Convergence_Indicator'] = calculate_convergence_indicator(df, MA_LABELS)
df['Decision'] = df.apply(enhanced_decision_rule, axis=1)

# ---------------------------
# MUSTERERKENNUNG DURCHFÜHREN
# ---------------------------

def detect_buy_sell_signals(df):
    signals = []
    last_buy_price = None
    last_sell_price = None
    holding = False
    for i in range(max(MA_WINDOWS), len(df)):
        current_price = df['Preis'].iloc[i]
        date = df['Datum'].iloc[i]
        
        # Überprüfen, ob wir aktuell halten oder nicht
        if not holding and last_sell_price is not None:
            delta_p = (current_price - last_sell_price) / last_sell_price
            
            # Bedingung für Preisrückgang von MIN_PRICE_DROP_1
            if delta_p <= MIN_PRICE_DROP_1:
                # Überprüfen, ob der Abwärtstrend durchbrochen wurde (MA_CROSS_SHORT)
                ma_short, ma_long = MA_CROSS_SHORT
                if df[f'MA {ma_short}'].iloc[i] > df[f'MA {ma_long}'].iloc[i] and \
                   df[f'MA {ma_short}'].iloc[i - 1] <= df[f'MA {ma_long}'].iloc[i - 1]:
                    # Zusätzliche Bedingungen prüfen
                    ma_conditions = all(
                        (df[f'MA {n}'].iloc[i] - df[f'MA {n}'].iloc[i - 1]) / df[f'MA {n}'].iloc[i - 1] < 0
                        for n in [ma_short, ma_long, 50, 100]
                    )
                    # Überprüfen, ob MA_CROSS_MEDIUM innerhalb von MA_CROSS_DAYS Tagen erfolgt
                    ma_medium_short, ma_medium_long = MA_CROSS_MEDIUM
                    ma_cross_occurred = False
                    for j in range(i, min(i + MA_CROSS_DAYS, len(df))):
                        if df[f'MA {ma_medium_short}'].iloc[j] > df[f'MA {ma_medium_long}'].iloc[j] and \
                           df[f'MA {ma_medium_short}'].iloc[j - 1] <= df[f'MA {ma_medium_long}'].iloc[j - 1]:
                            ma_cross_occurred = True
                            break
                    if ma_conditions and ma_cross_occurred:
                        signals.append((date, 'Kauf'))
                        last_buy_price = current_price
                        holding = True
                        continue  # Weiter zum nächsten Tag

            # Bedingung für Preisrückgang von MIN_PRICE_DROP_2
            if delta_p <= MIN_PRICE_DROP_2:
                if df[f'MA {ma_short}'].iloc[i] > df[f'MA {ma_long}'].iloc[i] and \
                   df[f'MA {ma_short}'].iloc[i - 1] <= df[f'MA {ma_long}'].iloc[i - 1]:
                    signals.append((date, 'Kauf'))
                    last_buy_price = current_price
                    holding = True
                    continue

            # Bedingung für Preisrückgang von MIN_PRICE_DROP_3
            if delta_p <= MIN_PRICE_DROP_3:
                if current_price > df[f'MA {ma_short}'].iloc[i] and \
                   df['Preis'].iloc[i - 1] <= df[f'MA {ma_short}'].iloc[i - 1]:
                    signals.append((date, 'Kauf'))
                    last_buy_price = current_price
                    holding = True
                    continue

        elif holding:
            # Verkaufsbedingungen
            delta_profit = (current_price - last_buy_price) / last_buy_price
            if delta_profit >= MIN_PROFIT:
                # Verkaufs-MA-Kreuzung überprüfen
                sell_ma_short, sell_ma_long = SELL_MA_CROSS
                if df[f'MA {sell_ma_short}'].iloc[i] < df[f'MA {sell_ma_long}'].iloc[i] and \
                   df[f'MA {sell_ma_short}'].iloc[i - 1] >= df[f'MA {sell_ma_long}'].iloc[i - 1]:
                    signals.append((date, 'Verkauf'))
                    last_sell_price = current_price
                    last_buy_price = None
                    holding = False
                    continue
        else:
            continue
    return signals

signals = detect_buy_sell_signals(df)

# ---------------------------
# GEWINNBERECHNUNG
# ---------------------------

def calculate_profit(df, signals, investitionsbetrag):
    balance = investitionsbetrag
    btc_holding = 0
    last_action = None
    last_buy_price = None
    trade_history = []
    
    for date, signal in signals:
        preis = df.loc[df['Datum'] == date, 'Preis'].values[0]
        if signal == 'Kauf' and last_action != 'Kauf':
            # Kaufen
            btc_holding = balance / preis
            balance = 0
            last_action = 'Kauf'
            last_buy_price = preis
            trade_history.append({'Datum': date, 'Aktion': 'Kauf', 'Preis': preis, 'BTC': btc_holding, 'Balance': balance})
        elif signal == 'Verkauf' and last_action == 'Kauf' and preis > last_buy_price:
            # Verkaufen nur, wenn der Preis höher ist als der letzte Kaufpreis
            balance = btc_holding * preis
            btc_holding = 0
            last_action = 'Verkauf'
            last_buy_price = None
            trade_history.append({'Datum': date, 'Aktion': 'Verkauf', 'Preis': preis, 'BTC': btc_holding, 'Balance': balance})

    # Am Ende alles verkaufen, falls noch BTC gehalten werden und der Preis höher ist als der letzte Kaufpreis
    if btc_holding > 0:
        preis = df['Preis'].iloc[-1]
        if preis > last_buy_price:
            balance = btc_holding * preis
            trade_history.append({'Datum': df['Datum'].iloc[-1], 'Aktion': 'Verkauf (Ende)', 'Preis': preis, 'BTC': 0, 'Balance': balance})
            btc_holding = 0
        else:
            # Wenn der aktuelle Preis niedriger ist als der letzte Kaufpreis, behalten wir die BTC
            balance = btc_holding * last_buy_price  # Wir bewerten zum letzten Kaufpreis
            trade_history.append({'Datum': df['Datum'].iloc[-1], 'Aktion': 'Halten', 'Preis': preis, 'BTC': btc_holding, 'Balance': balance})

    gesamtgewinn = balance - investitionsbetrag
    return gesamtgewinn, trade_history

# ---------------------------
# HAUPTFUNKTION
# ---------------------------

def main():
    investitionsbetrag = INVESTITIONSBETRAG

    # Gewinnberechnung
    gesamtgewinn, trade_history = calculate_profit(df, signals, investitionsbetrag)
    print(f"\nGesamtgewinn: {gesamtgewinn:.2f} USD")

    # ---------------------------
    # Visualisierung
    # ---------------------------

    # Subplots erstellen
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            'BTC Kurs mit MAs und Signalen',
            'Neue Indikatoren',
            'Entscheidungen',
            'Abstände zwischen MAs'
        )
    )

    # Erste Subplot: BTC Kurs und MAs
    fig.add_trace(
        go.Scatter(x=df['Datum'], y=df['Preis'], mode='lines', name='BTC Kurs', line=dict(color='white')),
        row=1, col=1
    )
    for label, color in zip(MA_LABELS, MA_COLORS):
        fig.add_trace(
            go.Scatter(x=df['Datum'], y=df[label], mode='lines', name=label, line=dict(color=color)),
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
                hovertemplate=f'{signal}: {date.strftime("%d.%m.%Y")}'
            ),
            row=1, col=1
        )

    # Zweite Subplot: Neue Indikatoren
    fig.add_trace(
        go.Scatter(x=df['Datum'], y=df['MA_Distance'], mode='lines', name='MA Distanz'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Datum'], y=df['Breakthrough_Signal'], mode='lines', name='Durchbruchsignal'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Datum'], y=df['Convergence_Indicator'], mode='lines', name='Konvergenzindikator'),
        row=2, col=1
    )

    # Dritte Subplot: Entscheidungen
    decision_colors = {'Ausbruch erwartet': 'yellow', 'Kauf erwägen': 'green', 'Verkauf erwägen': 'red', 'Neutral': 'gray'}
    for decision in decision_colors:
        mask = df['Decision'] == decision
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, 'Datum'],
                y=df.loc[mask, 'Preis'],
                mode='markers',
                name=decision,
                marker=dict(color=decision_colors[decision], size=8),
                showlegend=True
            ),
            row=3, col=1
        )

    # Vierte Subplot: Abstände zwischen MAs
    for dist_label in distance_columns:
        fig.add_trace(
            go.Scatter(
                x=df['Datum'],
                y=df[dist_label],
                mode='lines',
                name=dist_label,
                line=dict(width=1),
                visible='legendonly'
            ),
            row=4, col=1
        )

    # Layout anpassen
    fig.update_layout(
        height=1600,
        title='Erweiterte BTC Kursanalyse mit neuen Indikatoren und angepasster Strategie',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(
            title='Legende',
            orientation='v',
            x=1.02,
            y=1,
            bordercolor='white',
            borderwidth=1,
        ),
    )

    # Gewinnanzeige hinzufügen
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.5, y=-0.1,
        text=f"Investitionsbetrag: {investitionsbetrag:.2f} USD<br>Gesamtgewinn: {gesamtgewinn:.2f} USD",
        showarrow=False,
        font=dict(size=14, color='white'),
        align='center'
    )

    # Responsives Design aktivieren
    config = {'responsive': True}

    # Diagramm anzeigen
    fig.show(config=config)

if __name__ == "__main__":
    main()
