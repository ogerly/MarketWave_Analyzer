import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State, callback_context, dash_table
import dash
import plotly.express as px

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

df = get_daily_data()

# Berechnung der gleitenden Durchschnitte
ma_windows = [9, 20, 50, 100, 200, 400]
ma_colors = ['yellow', 'red', 'orange', 'green', 'purple', 'turquoise']
ma_labels = [f'MA {window}' for window in ma_windows]
for window, label in zip(ma_windows, ma_labels):
    df[label] = df['Preis'].rolling(window=window).mean()

# ---------------------------
# DASH APP EINRICHTUNG
# ---------------------------

app = Dash(__name__)

# ---------------------------
# LAYOUT DER APP
# ---------------------------

app.layout = html.Div([
    html.H1("BTC Analyse Tool mit interaktiven Kauf- und Verkaufspunkten"),
    html.Div([
        html.Label("Investitionsbetrag (USD):"),
        dcc.Input(id='investment-input', type='number', value=1000, min=0),
    ], style={'margin-bottom': '20px'}),
    dcc.Graph(id='price-chart', config={'displayModeBar': True}),
    html.Div([
        html.H2("Trade-Informationen"),
        dash_table.DataTable(
            id='trade-table',
            columns=[
                {'name': 'Datum', 'id': 'Datum'},
                {'name': 'Aktion', 'id': 'Aktion'},
                {'name': 'Preis', 'id': 'Preis'},
                {'name': 'Menge', 'id': 'Menge'},
                {'name': 'Balance', 'id': 'Balance'},
                {'name': 'Gewinn', 'id': 'Gewinn'},
            ],
            data=[],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
        ),
        html.Div(id='total-profit', style={'fontSize': '24px', 'fontWeight': 'bold', 'marginTop': '20px'}),
    ]),
])

# ---------------------------
# CALLBACKS
# ---------------------------

# Speichern der Trades in einer globalen Variable
trades = []

@app.callback(
    [Output('price-chart', 'figure'),
     Output('trade-table', 'data'),
     Output('total-profit', 'children')],
    [Input('price-chart', 'relayoutData'),
     Input('investment-input', 'value')],
    [State('price-chart', 'figure')]
)
def update_trades(relayoutData, investment, figure):
    ctx = callback_context
    global trades

    # Sicherstellen, dass 'figure' nicht None ist
    if figure is None:
        figure = create_figure()

    # Wenn die Investition geändert wurde, Trades zurücksetzen
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'investment-input.value':
        trades = []
        fig = create_figure()
        return fig, [], f"Gesamtgewinn: $0.00"

    # Wenn keine Interaktion stattfand
    if relayoutData is None:
        fig = create_figure()
        return fig, [], f"Gesamtgewinn: $0.00"

    # Aktuelle Shapes aus dem relayoutData oder dem Figure Layout holen
    if 'shapes' in relayoutData:
        annotations = relayoutData['shapes']
    elif figure and 'layout' in figure and 'shapes' in figure['layout']:
        annotations = figure['layout']['shapes']
    else:
        annotations = []

    # Verarbeiten der neuen Annotationen
    fig = create_figure()

    # Extrahieren der Kauf- und Verkaufspunkte
    new_trades = []
    for shape in annotations:
        if shape['type'] == 'line':
            x = pd.to_datetime(shape['x0']).date()
            y = shape['y0']
            color = shape['line']['color']
            if color == 'green':
                action = 'Kauf'
            elif color == 'red':
                action = 'Verkauf'
            else:
                continue  # Ignoriere Linien anderer Farben
            new_trades.append({'Datum': x, 'Aktion': action, 'Preis': y})

    # Sortieren nach Datum
    new_trades = sorted(new_trades, key=lambda x: x['Datum'])

    # Berechnungen durchführen
    trade_history, total_profit = calculate_trade_history(new_trades, investment)

    # Aktualisieren der Markierungen im Diagramm
    for trade in new_trades:
        color = 'green' if trade['Aktion'] == 'Kauf' else 'red'
        symbol = 'triangle-up' if trade['Aktion'] == 'Kauf' else 'triangle-down'
        fig.add_trace(go.Scatter(
            x=[trade['Datum']],
            y=[trade['Preis']],
            mode='markers',
            marker=dict(symbol=symbol, color=color, size=12),
            name=trade['Aktion'],
            showlegend=False
        ))

    # Annotationen hinzufügen
    shapes = []
    for trade in new_trades:
        color = 'green' if trade['Aktion'] == 'Kauf' else 'red'
        shapes.append({
            'type': 'line',
            'x0': trade['Datum'],
            'y0': trade['Preis'],
            'x1': trade['Datum'],
            'y1': trade['Preis'],
            'line': {
                'color': color,
                'width': 3,
            },
            'xref': 'x',
            'yref': 'y',
        })
    fig.update_layout(shapes=shapes)

    # Tabelle aktualisieren
    table_data = [{
        'Datum': trade['Datum'].strftime('%Y-%m-%d'),
        'Aktion': trade['Aktion'],
        'Preis': f"${trade['Preis']:.2f}",
        'Menge': f"{trade['Menge']:.6f} BTC",
        'Balance': f"${trade['Balance']:.2f}",
        'Gewinn': f"${trade['Gewinn']:.2f}",
    } for trade in trade_history]

    return fig, table_data, f"Gesamtgewinn: ${total_profit:.2f}"

def create_figure():
    # Erstellen des Basisdiagramms
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Datum'], y=df['Preis'], mode='lines', name='BTC Kurs', line=dict(color='lightgray'), opacity=0.6))
    for label, color in zip(ma_labels, ma_colors):
        fig.add_trace(go.Scatter(x=df['Datum'], y=df[label], mode='lines', name=label, line=dict(color=color)))

    # Layout anpassen
    fig.update_layout(
        title='BTC Kurs mit gleitenden Durchschnitten',
        xaxis_title='Datum',
        yaxis_title='Preis (USD)',
        hovermode='x',
        dragmode='drawline',  # Ermöglicht das Zeichnen von Linien
        newshape=dict(line_color='green'),
    )

    # Zeichnen einschränken auf Linien
    fig.update_layout(
        shapes=[],
        modebar_add=['drawline', 'eraseshape']
    )

    return fig

def calculate_trade_history(trades, initial_investment):
    balance = initial_investment
    btc_holding = 0
    trade_history = []
    total_profit = 0

    for trade in trades:
        date = trade['Datum']
        preis = trade['Preis']
        action = trade['Aktion']

        if action == 'Kauf':
            if balance <= 0:
                # Kein verfügbares Kapital
                continue
            # Maximal verfügbares Kapital investieren
            investment_amount = balance
            btc_amount = investment_amount / preis
            btc_holding += btc_amount
            balance -= investment_amount
            trade_history.append({
                'Datum': date,
                'Aktion': 'Kauf',
                'Preis': preis,
                'Menge': btc_amount,
                'Balance': balance,
                'Gewinn': 0.0,
            })
        elif action == 'Verkauf' and btc_holding > 0:
            # Alle BTC verkaufen
            proceeds = btc_holding * preis
            profit = proceeds - initial_investment
            balance += proceeds
            total_profit += profit
            trade_history.append({
                'Datum': date,
                'Aktion': 'Verkauf',
                'Preis': preis,
                'Menge': -btc_holding,
                'Balance': balance,
                'Gewinn': profit,
            })
            btc_holding = 0
        else:
            continue

    return trade_history, total_profit

# ---------------------------
# APP AUSFÜHREN
# ---------------------------

if __name__ == '__main__':
    app.run_server(debug=True)
