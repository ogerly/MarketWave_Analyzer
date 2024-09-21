# MarketWave Analyzer
### Kursanalyse mit gleitenden Durchschnitten

Dieses Python-Skript analysiert historische Bitcoin-Kursdaten mithilfe von gleitenden Durchschnitten (Moving Averages, MAs). Es erkennt Muster wie Bodenzonen, Spitzen, Auf- und Abwärtstrends sowie Kauf- und Verkaufsignale.

## Funktionen

1. **Datenverarbeitung**: 
   - Liest BTC-USD Tageskurse aus einer CSV-Datei
   - Berechnet gleitende Durchschnitte (9, 20, 50, 100, 200, 400 Tage)

2. **Mustererkennung**:
   - `detect_bottoms()`: Identifiziert potenzielle Bodenzonen
   - `detect_peaks()`: Erkennt Kursspitzen
   - `detect_uptrends()` und `detect_downtrends()`: Analysieren Trendphasen
   - `detect_buy_sell_signals()`: Generiert Kauf- und Verkaufssignale

3. **Gewinnberechnung**:
   - Simuliert Trades basierend auf den erkannten Signalen
   - Berechnet den potenziellen Gewinn für einen gegebenen Investitionsbetrag

4. **Visualisierung**:
   - Erstellt ein interaktives Plotly-Diagramm mit zwei Untergrafiken:
     1. BTC-Kurs mit gleitenden Durchschnitten, Bodenzonen, Spitzen und Handelssignalen
     2. Flächendiagramm der Abstände zwischen den MAs

## Verwendung

1. Stellen Sie sicher, dass die erforderlichen Bibliotheken installiert sind: 
   ```
   pip install pandas plotly
   ```

2. Legen Sie die BTC-USD Tageskursdaten als 'BTC__USD_daily.csv' im gleichen Verzeichnis ab.

3. Führen Sie das Skript aus und geben Sie den gewünschten Investitionsbetrag ein.

4. Das Skript generiert ein interaktives Diagramm und gibt den berechneten Gesamtgewinn aus.

## Anpassungsmöglichkeiten

- Passen Sie die `ma_windows`-Liste an, um andere gleitende Durchschnitte zu verwenden
- Ändern Sie den `distance_threshold` für die Bodenerkennung
- Modifizieren Sie die Funktionen zur Mustererkennung für andere Strategien

## Hinweise

- Dies ist ein Analysewerkzeug und keine Anlageempfehlung
- Vergangene Performance garantiert keine zukünftigen Ergebnisse
- Berücksichtigen Sie stets Risiken und führen Sie eigene Recherchen durch, bevor Sie investieren

Möchten Sie, dass ich noch weitere Details zum Code oder zur Funktionsweise hinzufüge?