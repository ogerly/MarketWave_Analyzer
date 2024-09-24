# MarketWave Analyzer

### Interaktive Kursanalyse mit gleitenden Durchschnitten

**MarketWave Analyzer** ist ein fortschrittliches Python-Tool zur Analyse historischer Bitcoin-Kursdaten unter Verwendung von gleitenden Durchschnitten (Moving Averages, MAs). Es bietet umfangreiche Funktionen zur Erkennung von Mustern wie Bodenzonen, Spitzen, Trendphasen sowie zur Generierung von Kauf- und Verkaufssignalen. Durch die Integration von interaktiven Elementen ermöglicht es Benutzern, Handelsstrategien zu simulieren und die Auswirkungen von Entscheidungen in Echtzeit zu beobachten.

![MarketWave Analyzer Screenshot](https://github.com/user-attachments/assets/c09f979e-6103-4ec0-8bc4-66fa8c59b7c2)

## Funktionen

1. **Datenverarbeitung**:
   - **Flexible Datenquellen**: Unterstützung für tägliche und stündliche BTC-USD-Kursdaten.
   - **Gleitende Durchschnitte**: Berechnung von MAs über benutzerdefinierte Zeiträume.
   - **Interaktive Parametereingabe**: Anpassung von Analyseparametern während der Laufzeit.

2. **Mustererkennung und Indikatoren**:
   - **Erkennung von Trends**: Identifizierung von Auf- und Abwärtstrends basierend auf MAs.
   - **Indikatoren**:
     - **MA Distance**: Misst die durchschnittliche Distanz zwischen allen MAs.
     - **Breakthrough Signal**: Erkennt Kreuzungen zwischen verschiedenen MAs.
     - **Convergence Indicator**: Bewertet die Konvergenz der MAs.

3. **Interaktive Handelsfunktionalität**:
   - **Manuelles Setzen von Kauf- und Verkaufspunkten**: Direkt im Diagramm durch Zeichnen von Linien.
   - **Verschieben und Löschen von Punkten**: Anpassung der Handelsentscheidungen in Echtzeit.
   - **Live-Berechnung von Handelsgewinnen**: Aktualisierung von Kontostand, BTC-Bestand und Gewinnen bei jeder Änderung.

4. **Gewinnberechnung**:
   - **Simulierte Trades**: Basierend auf den gesetzten Kauf- und Verkaufspunkten.
   - **Dynamische Kapitalverwaltung**: Investition des verfügbaren Kapitals bei jedem Kauf.
   - **Gewinn- und Verlustrechnung**: Übersichtliche Darstellung pro Trade und kumulativ.

5. **Visualisierung**:
   - **Interaktives Diagramm**:
     - Darstellung des BTC-Kurses mit gleitenden Durchschnitten.
     - Markierung von Kauf- und Verkaufspunkten.
     - Visualisierung der Indikatoren und Entscheidungsbereiche.
   - **Anpassbares Layout**: Möglichkeit, verschiedene Indikatoren und Zeiträume zu visualisieren.

## Installation

1. **Erforderliche Bibliotheken installieren**:

   ```bash
   pip install pandas numpy requests plotly dash
   ```

2. **Daten bereitstellen**:

   - Laden Sie die BTC-USD Tageskursdaten herunter und speichern Sie sie als `BTC__USD_daily.csv` im gleichen Verzeichnis wie das Skript.

3. **Skript herunterladen**:

   - Laden Sie das Skript `btc_analysis_tool.py` aus dem Repository herunter.

## Verwendung

1. **Anwendung starten**:

   ```bash
   python btc_analysis_tool.py
   ```

   - Die Dash-Anwendung startet und gibt eine URL aus (z.B. `http://127.0.0.1:8050/`).
   - Öffnen Sie diese URL in Ihrem Webbrowser.

2. **Investitionsbetrag festlegen**:

   - Geben Sie im Feld "Investitionsbetrag (USD)" den Betrag ein, den Sie investieren möchten.

3. **Kauf- und Verkaufspunkte setzen**:

   - **Kaufpunkt setzen**:
     - Wählen Sie in der Werkzeugleiste des Diagramms das Linienzeichnungswerkzeug.
     - Zeichnen Sie eine vertikale Linie an der Stelle, an der Sie kaufen möchten.
     - Stellen Sie sicher, dass die Linie **grün** ist (Farbe kann in den Einstellungen des Zeichentools angepasst werden).

   - **Verkaufspunkt setzen**:
     - Zeichnen Sie eine vertikale Linie an der Stelle, an der Sie verkaufen möchten.
     - Stellen Sie sicher, dass die Linie **rot** ist.

   - **Punkte verschieben oder löschen**:
     - Um einen Punkt zu verschieben, klicken Sie auf die Linie und ziehen Sie sie an die gewünschte Position.
     - Um einen Punkt zu löschen, wählen Sie das Radiergummi-Werkzeug und klicken Sie auf die Linie.

4. **Handelsinformationen beobachten**:

   - **Trade-Informationen**:
     - Unter dem Diagramm wird eine Tabelle mit allen Trades angezeigt, inklusive Datum, Aktion, Preis, Menge, Balance und Gewinn.
   - **Gesamtgewinn**:
     - Der kumulative Gewinn aller Trades wird unter der Tabelle angezeigt und aktualisiert sich bei jeder Änderung.

5. **Anpassung der Parameter** (optional):

   - Sie können die MA-Perioden und andere Analyseparameter direkt im Code anpassen oder das Skript erweitern, um zusätzliche Eingabemöglichkeiten zu bieten.

## Anpassungsmöglichkeiten

- **Gleitende Durchschnitte**:
  - Passen Sie die Liste `ma_windows` im Code an, um andere Zeiträume für die MAs zu verwenden.

- **Indikatoren und Entscheidungsregeln**:
  - Modifizieren Sie die Schwellenwerte und Berechnungen in den Funktionen `calculate_ma_distance`, `calculate_breakthrough_signal` und `enhanced_decision_rule`, um die Analyse an Ihre Bedürfnisse anzupassen.

- **Visualisierung**:
  - Fügen Sie weitere Diagramme oder Indikatoren hinzu, um zusätzliche Einblicke zu erhalten.

- **Datenquellen**:
  - Implementieren Sie die Funktion `get_hourly_data` mit einer gültigen API, um stündliche Daten zu verwenden.

## Wichtige Hinweise

- **Keine Anlageberatung**:
  - Dieses Tool dient ausschließlich zu Analysezwecken und stellt keine Empfehlung zum Kauf oder Verkauf von Kryptowährungen dar.

- **Risiken beachten**:
  - Der Handel mit Kryptowährungen ist mit hohen Risiken verbunden. Vergangene Performance ist kein Indikator für zukünftige Ergebnisse.

- **Eigenverantwortung**:
  - Bitte führen Sie eigene Recherchen durch und konsultieren Sie bei Bedarf einen Finanzberater, bevor Sie Handelsentscheidungen treffen.

## Fehlersuche und Unterstützung

- **Bekannte Probleme**:
  - **Fehlermeldung bezüglich `NoneType`**: Stellen Sie sicher, dass die `figure`-Variable korrekt initialisiert ist (siehe Codeanpassungen in den letzten Abschnitten).
  - **Veraltete Pakete**: Aktualisieren Sie Ihre Dash-Version und passen Sie die Import-Anweisungen an (`from dash import dash_table`).

- **Unterstützung**:
  - Bei Fragen oder Problemen können Sie sich an den Entwickler wenden oder ein Issue im Repository erstellen.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Weitere Informationen finden Sie in der `LICENSE`-Datei.

## Danksagung

Vielen Dank an alle Mitwirkenden und Tester, die zur Entwicklung dieses Tools beigetragen haben.

---

# Kryptowährungshandel mit Moving Averages und Breakthrough Signals

Dieses Projekt bietet eine interaktive Analyseplattform für Kryptowährungen, die auf Moving Averages (MAs) und Breakthrough Signals basiert. Die Plattform ermöglicht es Benutzern, Trades zu setzen, zu verschieben oder zu löschen, und bietet eine visuelle Darstellung der Analyseergebnisse.

## Installation

1. **Python 3.8 oder höher**:
   - Stellen Sie sicher, dass Sie Python 3.8 oder höher installiert haben. Weitere Informationen finden Sie unter <https://www.python.org/downloads/>.