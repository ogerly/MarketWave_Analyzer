import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------
# Anpassbare Parameter
# ---------------------------

# Liste der gleitenden Durchschnitte und Farben
ma_windows = [9, 20, 50, 100, 200, 400]
ma_colors = ['yellow', 'red', 'orange', 'green', 'purple', 'turquoise']
ma_labels = [f'MA {window}' for window in ma_windows]

# Schwellenwert für minimale Abstände zwischen den MAs bei der Bodenerkennung
distance_threshold = 0.005  # Kann angepasst werden

# ---------------------------
# Daten einlesen und vorbereiten
# ---------------------------

columns = ['Datum', 'Zuletzt', 'Eröffn.', 'Hoch', 'Tief', 'Vol.', '+/- %']
url = 'BTC__USD_daily.csv'
df = pd.read_csv(url, skiprows=1, names=columns)
df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y')
df['Zuletzt'] = df['Zuletzt'].str.replace('.', '').str.replace(',', '.').astype(float)
df = df.iloc[::-1].reset_index(drop=True)

# Den tatsächlichen BTC-Kurs hinzufügen
df['Preis'] = df['Zuletzt']

# Berechnung der gleitenden Durchschnitte
for window, label in zip(ma_windows, ma_labels):
    df[label] = df['Preis'].rolling(window=window).mean()

# Abstände zwischen den MAs berechnen
df['Dist_MA9_MA20'] = df['MA 9'] - df['MA 20']
df['Dist_MA20_MA50'] = df['MA 20'] - df['MA 50']
df['Dist_MA50_MA100'] = df['MA 50'] - df['MA 100']
df['Dist_MA100_MA200'] = df['MA 100'] - df['MA 200']
df['Dist_MA200_MA400'] = df['MA 200'] - df['MA 400']

# ---------------------------
# Funktionen zur Mustererkennung
# ---------------------------

# Funktion zur Erkennung von Bodenzonen
def detect_bottoms(df):
    bottoms = []
    for i in range(1, len(df)):
        # Prüfen, ob die Abstände zwischen den MAs minimal sind
        distances = [
            abs(df['MA 9'].iloc[i] - df['MA 20'].iloc[i]),
            abs(df['MA 20'].iloc[i] - df['MA 50'].iloc[i]),
            abs(df['MA 50'].iloc[i] - df['MA 100'].iloc[i]),
            abs(df['MA 100'].iloc[i] - df['MA 200'].iloc[i]),
            abs(df['MA 200'].iloc[i] - df['MA 400'].iloc[i]) if 'MA 400' in df.columns else 0
        ]
        if all(dist / df['Preis'].iloc[i] < distance_threshold for dist in distances):
            # Prüfen, ob MA20 die höheren MAs von unten nach oben durchbricht
            ma20_crossed_above = all(
                df['MA 20'].iloc[i] > df[ma].iloc[i] and df['MA 20'].iloc[i - 1] <= df[ma].iloc[i - 1]
                for ma in ['MA 50', 'MA 100', 'MA 200', 'MA 400'] if ma in df.columns
            )
            # Prüfen, ob MA50 die höheren MAs von unten nach oben durchbricht
            ma50_crossed_above = all(
                df['MA 50'].iloc[i] > df[ma].iloc[i] and df['MA 50'].iloc[i - 1] <= df[ma].iloc[i - 1]
                for ma in ['MA 100', 'MA 200', 'MA 400'] if ma in df.columns
            )
            if ma20_crossed_above and ma50_crossed_above:
                bottoms.append(df['Datum'].iloc[i])
    return bottoms

# Funktion zur Erkennung von Spitzen
def detect_peaks(df):
    peaks = []
    i = 1
    while i < len(df):
        # Anfang der Spitze: MA9 kreuzt MA20 nach oben
        if df['MA 9'].iloc[i] > df['MA 20'].iloc[i] and df['MA 9'].iloc[i - 1] <= df['MA 20'].iloc[i - 1]:
            start_date = df['Datum'].iloc[i]
            # Suche nach dem Ende der Spitze
            for j in range(i+1, len(df)):
                if df['MA 20'].iloc[j] < df['MA 50'].iloc[j] and df['MA 20'].iloc[j - 1] >= df['MA 50'].iloc[j - 1]:
                    end_date = df['Datum'].iloc[j]
                    peaks.append((start_date, end_date))
                    i = j  # Fortfahren ab dem Ende der Spitze
                    break
            else:
                # Kein Ende der Spitze gefunden
                i += 1
        else:
            i += 1
    return peaks

# Funktion zur Erkennung von Aufwärtstrends
def detect_uptrends(df):
    uptrends = []
    in_uptrend = False
    for i in range(len(df)):
        if (df['MA 9'].iloc[i] > df['MA 20'].iloc[i] and
            df['MA 20'].iloc[i] > df['MA 50'].iloc[i] and
            df['MA 50'].iloc[i] > df['MA 100'].iloc[i]):
            if not in_uptrend:
                in_uptrend = True
                start_date = df['Datum'].iloc[i]
        else:
            if in_uptrend:
                in_uptrend = False
                end_date = df['Datum'].iloc[i]
                uptrends.append((start_date, end_date))
    if in_uptrend:
        uptrends.append((start_date, df['Datum'].iloc[-1]))
    return uptrends

# Funktion zur Erkennung von Abwärtstrends
def detect_downtrends(df):
    downtrends = []
    in_downtrend = False
    for i in range(len(df)):
        if (df['MA 9'].iloc[i] < df['MA 20'].iloc[i] and
            df['MA 20'].iloc[i] < df['MA 50'].iloc[i]):
            if not in_downtrend:
                in_downtrend = True
                start_date = df['Datum'].iloc[i]
        else:
            if in_downtrend:
                in_downtrend = False
                end_date = df['Datum'].iloc[i]
                downtrends.append((start_date, end_date))
    if in_downtrend:
        downtrends.append((start_date, df['Datum'].iloc[-1]))
    return downtrends

# Funktion zur Erkennung von Kauf- und Verkaufsignalen
def detect_buy_sell_signals(df):
    signals = []
    for i in range(1, len(df)):
        if df['MA 20'].iloc[i] > df['MA 50'].iloc[i] and df['MA 20'].iloc[i - 1] <= df['MA 50'].iloc[i - 1]:
            signals.append((df['Datum'].iloc[i], 'Kauf'))
        elif df['MA 20'].iloc[i] < df['MA 50'].iloc[i] and df['MA 20'].iloc[i - 1] >= df['MA 50'].iloc[i - 1]:
            signals.append((df['Datum'].iloc[i], 'Verkauf'))
    return signals

# ---------------------------
# Investitionsbetrag erfassen
# ---------------------------

# Eingabe des Investitionsbetrags
investitionsbetrag = float(input("Bitte geben Sie den Investitionsbetrag in USD ein: "))

# ---------------------------
# Mustererkennung durchführen
# ---------------------------

bottoms = detect_bottoms(df)
peaks = detect_peaks(df)
uptrends = detect_uptrends(df)
downtrends = detect_downtrends(df)
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

# Bodenzonen markieren
for date in bottoms:
    preis = df.loc[df['Datum'] == date, 'Preis'].values[0]
    fig.add_trace(
        go.Scatter(
            x=[date],
            y=[preis],
            mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=12),
            name='Boden',
            showlegend=False,
            hovertemplate=f'Boden: {date.strftime("%d.%m.%Y")}'
        ),
        row=1, col=1
    )

# Spitzen markieren
for start_date, end_date in peaks:
    preis_start = df.loc[df['Datum'] == start_date, 'Preis'].values[0]
    preis_end = df.loc[df['Datum'] == end_date, 'Preis'].values[0]
    # Anfang der Spitze
    fig.add_trace(
        go.Scatter(
            x=[start_date],
            y=[preis_start],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=12),
            name='Spitze Beginn',
            showlegend=False,
            hovertemplate=f'Spitze Beginn: {start_date.strftime("%d.%m.%Y")}'
        ),
        row=1, col=1
    )
    # Ende der Spitze
    fig.add_trace(
        go.Scatter(
            x=[end_date],
            y=[preis_end],
            mode='markers',
            marker=dict(symbol='triangle-down', color='orange', size=12),
            name='Spitze Ende',
            showlegend=False,
            hovertemplate=f'Spitze Ende: {end_date.strftime("%d.%m.%Y")}'
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
            hovertemplate=f'{signal}: {date.strftime("%d.%m.%Y")}'
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
