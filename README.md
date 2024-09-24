# Aktualisierte README für das MarketWave Analyzer Projekt

## MarketWave Analyzer

### Interaktive Kursanalyse mit gleitenden Durchschnitten und fortgeschrittenen Methoden

**MarketWave Analyzer** ist ein umfassendes Open-Source-Tool zur Analyse von Finanzmarktdaten, insbesondere von Bitcoin (BTC). Das Projekt zielt darauf ab, verschiedene statistische Methoden, Zeitreihenanalysen und maschinelles Lernen zu integrieren, um ein leistungsfähiges Analysewerkzeug zu schaffen. Die einzelnen Komponenten werden schrittweise entwickelt und zusammengeführt, um ein vollständiges Analyse-Tool zu bilden.

![MarketWave Analyzer Screenshot](https://github.com/ogerly/MarketWave_Analyzer/assets/c09f979e-6103-4ec0-8bc4-66fa8c59b7c2)

---

## Funktionen

1. **Datenverarbeitung**:
   - Unterstützung für tägliche und stündliche BTC-USD-Kursdaten.
   - Flexibles Einlesen von Daten aus CSV-Dateien und APIs (z.B. Binance API).

2. **Gleitende Durchschnitte und Indikatoren**:
   - Berechnung von gleitenden Durchschnitten (MAs) über anpassbare Zeiträume.
   - Berechnung von Abständen zwischen verschiedenen MAs.
   - Implementierung von Indikatoren wie MA Distance, Breakthrough Signal und Convergence Indicator.

3. **Mustererkennung**:
   - Erkennung von Bodenzonen und Spitzen.
   - Identifizierung von Aufwärts- und Abwärtstrends.
   - Generierung von Kauf- und Verkaufssignalen basierend auf MAs und anderen Indikatoren.

4. **Zeitreihenanalyse mit ARIMA**:
   - Verwendung von ARIMA-Modellen zur Prognose von Kursbewegungen.
   - Bestätigung von Handelssignalen durch statistische Modelle.

5. **Interaktive Handelsfunktionalität**:
   - Manuelles Setzen, Verschieben und Löschen von Kauf- und Verkaufspunkten im Diagramm.
   - Live-Berechnung von Handelsgewinnen und Aktualisierung von Kontoständen und Gewinnen in Echtzeit.

6. **Visualisierung**:
   - Interaktive Diagramme mit Plotly und Dash.
   - Darstellung von Kursdaten, MAs, Handelssignalen und Indikatoren.
   - Anpassbare Layouts und Responsive Design.

---

## Projektstruktur und Komponenten

### 1. `btc_analysis_tool.py`

- **Beschreibung**: Hauptanwendung zur interaktiven Analyse von BTC-Kursdaten mit gleitenden Durchschnitten und Handelsfunktionalität.
- **Funktionen**:
  - Interaktives Setzen von Kauf- und Verkaufspunkten.
  - Live-Berechnung von Handelsgewinnen.
  - Visualisierung von Kursdaten und Indikatoren.

### 2. `arima_test.py`

- **Beschreibung**: Skript zur Integration von ARIMA-Modellen in die Analyse.
- **Funktionen**:
  - Abrufen von stündlichen BTC-Daten über die Binance API.
  - Überprüfung der Stationarität von Zeitreihen.
  - Anpassung von ARIMA-Modellen und Prognose zukünftiger Kursbewegungen.
  - Bestätigung von Handelssignalen durch ARIMA-Prognosen.

### 3. `app.py`

- **Beschreibung**: Erweiterung des Analyse-Tools mit zusätzlichen Mustererkennungsfunktionen.
- **Funktionen**:
  - Erkennung von Bodenzonen, Spitzen, Aufwärts- und Abwärtstrends.
  - Generierung von Kauf- und Verkaufssignalen.
  - Berechnung von potenziellen Gewinnen aus den erkannten Signalen.
  - Visualisierung von Kursdaten, MAs und Abständen zwischen MAs.

### 4. `index.html`

- **Beschreibung**: Startseite des Projekts mit Informationen, Roadmap und Blog.
- **Funktionen**:
  - Vorstellung des Projekts und seiner Ziele.
  - Darstellung der aktuellen Funktionen und zukünftigen Entwicklungen.
  - Integration eines Blogs zur Dokumentation von Erkenntnissen und Fortschritten.
  - Verlinkung zum GitHub-Repository und Einladung zur Mitarbeit.

---

## Roadmap

Wir planen, die folgenden Komponenten und Funktionen schrittweise zu integrieren, um ein umfassendes Analyse-Tool zu erstellen:

- **Integration von ARIMA-Modellen**: Verbesserung der Zeitreihenanalyse durch statistische Methoden.
- **Implementierung von Machine Learning**: Einsatz von Algorithmen zur Mustererkennung und Vorhersage.
- **Entwicklung von neuronalen Netzwerken (LSTM)**: Nutzung von Deep Learning für Zeitreihenvorhersagen.
- **Sentiment-Analyse**: Analyse von Finanznachrichten zur Bewertung von Marktsentiment.
- **Fraktalanalyse**: Untersuchung der Fraktalstruktur von Finanzmärkten.
- **Optimierung mittels genetischer Algorithmen**: Automatische Anpassung von Parametern für technische Indikatoren.
- **Erweiterte Visualisierung**: Entwicklung von komplexen Datenvisualisierungen zur Unterstützung der Analyse.

---

## Verwendung

### Voraussetzungen

- **Python-Version**: 3.7 oder höher
- **Erforderliche Bibliotheken**:

  ```bash
  pip install pandas numpy requests plotly dash statsmodels
  ```

### Datenbereitstellung

- **Tägliche Daten**: Stellen Sie sicher, dass die Datei `BTC__USD_daily.csv` im gleichen Verzeichnis wie die Skripte vorhanden ist.
- **Stündliche Daten**: `arima_test.py` lädt die Daten automatisch über die Binance API.

### Ausführen der Anwendungen

#### `btc_analysis_tool.py`

1. Starten Sie die Anwendung:

   ```bash
   python btc_analysis_tool.py
   ```

2. Öffnen Sie die angegebene URL in Ihrem Webbrowser (z.B. `http://127.0.0.1:8050/`).

3. Verwenden Sie die interaktiven Funktionen, um Kauf- und Verkaufspunkte zu setzen und die Handelsgewinne zu analysieren.

#### `arima_test.py`

1. Führen Sie das Skript aus:

   ```bash
   python arima_test.py
   ```

2. Geben Sie den gewünschten Investitionsbetrag ein, wenn Sie dazu aufgefordert werden.

3. Das Skript generiert ein interaktives Diagramm mit den ARIMA-Prognosen und gibt den berechneten Gesamtgewinn aus.

#### `app.py`

1. Führen Sie das Skript aus:

   ```bash
   python app.py
   ```

2. Geben Sie den gewünschten Investitionsbetrag ein, wenn Sie dazu aufgefordert werden.

3. Das Skript generiert ein interaktives Diagramm mit den erkannten Mustern und gibt den berechneten Gesamtgewinn aus.

---

## Anpassungsmöglichkeiten

- **Gleitende Durchschnitte**:
  - Passen Sie die Liste `ma_windows` in den Skripten an, um andere Zeiträume für die MAs zu verwenden.

- **Indikatoren und Entscheidungsregeln**:
  - Modifizieren Sie die Schwellenwerte und Berechnungen in den entsprechenden Funktionen, um die Analyse an Ihre Bedürfnisse anzupassen.

- **Visualisierung**:
  - Fügen Sie weitere Diagramme oder Indikatoren hinzu, um zusätzliche Einblicke zu erhalten.

- **Datenquellen**:
  - Passen Sie die Datenbeschaffung an, um andere Kryptowährungen oder Finanzinstrumente zu analysieren.

---

## Blog und Neueste Erkenntnisse

Wir haben unsere neuesten Erkenntnisse und Fortschritte in unserem **Blog** dokumentiert. Hier teilen wir unsere Erfahrungen, diskutieren Herausforderungen und präsentieren Lösungen.

- **Aktuelle Beiträge**:
  - **Integration von ARIMA-Modellen**: Erfahrungen bei der Prognose von BTC-Kursen.
  - **Interaktive Handelsfunktionen**: Entwicklung von Funktionen zum Setzen von Kauf- und Verkaufspunkten.
  - **Fehlerbehebung und Optimierung**: Umgang mit Fehlermeldungen und Aktualisierung von Paketen.

Wir werden den Blog regelmäßig mit neuen Erkenntnissen erweitern.

---

## Wichtige Hinweise

- **Keine Anlageberatung**:
  - Dieses Tool dient ausschließlich zu Analysezwecken und stellt keine Empfehlung zum Kauf oder Verkauf von Kryptowährungen dar.

- **Risiken beachten**:
  - Der Handel mit Kryptowährungen ist mit hohen Risiken verbunden. Vergangene Performance ist kein Indikator für zukünftige Ergebnisse.

- **Eigenverantwortung**:
  - Bitte führen Sie eigene Recherchen durch und konsultieren Sie bei Bedarf einen Finanzberater, bevor Sie Handelsentscheidungen treffen.

---

## Open Source auf GitHub

MarketWave Analyzer ist ein kollaboratives Projekt. Wir laden Entwickler, Datenwissenschaftler und Finanzexperten zur Mitarbeit ein.

- **Repository**: [GitHub: MarketWave Analyzer](https://github.com/ogerly/MarketWave_Analyzer)
- **Beiträge**:
  - Erkunden Sie unseren aktuellen Code.
  - Tragen Sie zur Weiterentwicklung bei.
  - Diskutieren Sie Ideen und zukünftige Funktionen.
  - Helfen Sie bei der Dokumentation und dem Testen.

Für Kontakt und Vorschläge erstellen Sie bitte ein Issue in unserem GitHub-Repository.

---

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Weitere Informationen finden Sie in der `LICENSE`-Datei.

---

 

 