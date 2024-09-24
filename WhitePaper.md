# Entwicklung eines interaktiven Analyse-Tools für Bitcoin-Kursdaten unter Verwendung gleitender Durchschnitte und statistischer Modelle

## Zusammenfassung

In diesem Papier stellen wir die Entwicklung eines interaktiven Analyse-Tools für Bitcoin (BTC) vor, das auf der Verwendung von gleitenden Durchschnitten (Moving Averages, MAs) und statistischen Modellen wie ARIMA basiert. Ziel ist es, Muster in historischen Kursdaten zu identifizieren, Handelssignale zu generieren und die potenziellen Gewinne zu simulieren. Wir erläutern die Methoden, Formeln und Algorithmen, die zur Mustererkennung und Gewinnberechnung verwendet wurden, und diskutieren die Ergebnisse unserer Analyse.

---

## Inhaltsverzeichnis

1. [Einleitung](#einleitung)
2. [Methodik](#methodik)
   - 2.1. Datenerfassung und -vorbereitung
   - 2.2. Berechnung der gleitenden Durchschnitte
   - 2.3. Indikatoren und Mustererkennung
   - 2.4. Zeitreihenanalyse mit ARIMA
   - 2.5. Gewinnberechnung
3. [Ergebnisse](#ergebnisse)
4. [Diskussion](#diskussion)
5. [Schlussfolgerungen und Ausblick](#schlussfolgerungen-und-ausblick)
6. [Literaturverzeichnis](#literaturverzeichnis)

---

## 1. Einleitung

Die Analyse von Finanzzeitreihen, insbesondere von Kryptowährungen wie Bitcoin, ist für Investoren und Analysten von großer Bedeutung. Technische Indikatoren wie gleitende Durchschnitte werden häufig verwendet, um Trends zu identifizieren und Handelsentscheidungen zu treffen. In diesem Papier präsentieren wir ein interaktives Analyse-Tool, das die Berechnung verschiedener gleitender Durchschnitte, die Erkennung von Mustern und die Integration statistischer Modelle zur Prognose zukünftiger Kursbewegungen ermöglicht.

---

## 2. Methodik

### 2.1. Datenerfassung und -vorbereitung

Wir verwenden historische BTC-USD-Kursdaten, die entweder aus einer CSV-Datei (`BTC__USD_daily.csv`) oder über die Binance API für stündliche Daten abgerufen werden. Die Daten werden wie folgt vorverarbeitet:

- **Zeitstempel**: Konvertierung der Datum- und Zeitangaben in ein geeignetes Format (`datetime`).
- **Preisinformationen**: Umwandlung der Kursdaten in Fließkommazahlen und Bereinigung von Sonderzeichen.
- **Sortierung**: Chronologische Anordnung der Datenreihen.

### 2.2. Berechnung der gleitenden Durchschnitte

Gleitende Durchschnitte (MAs) werden berechnet, um Trends und Muster in den Kursdaten zu erkennen. Wir verwenden verschiedene Zeitfenster \( n \) für die MAs, definiert durch die Liste \( n \in \{9, 20, 50, 100, 200, 400\} \).

Die Berechnung des gleitenden Durchschnitts \( MA_n \) über \( n \) Perioden erfolgt durch:

\[
MA_n(t) = \frac{1}{n} \sum_{i=0}^{n-1} P(t - i)
\]

wobei \( P(t) \) der Preis zum Zeitpunkt \( t \) ist.

### 2.3. Indikatoren und Mustererkennung

#### 2.3.1. Abstände zwischen gleitenden Durchschnitten

Die Abstände zwischen den MAs werden berechnet, um Konvergenzen oder Divergenzen zu identifizieren:

\[
\text{Dist}_{MA_i, MA_j}(t) = MA_i(t) - MA_j(t)
\]

#### 2.3.2. Erkennung von Bodenzonen

Eine Bodenzone wird erkannt, wenn die Abstände zwischen den MAs minimal sind und bestimmte Kreuzungen stattfinden.

**Kriterien**:

- Alle Abstände \( \text{Dist}_{MA_i, MA_{i+1}}(t) \) sind kleiner als ein Schwellenwert \( \delta \) relativ zum Preis \( P(t) \):

\[
\frac{|\text{Dist}_{MA_i, MA_{i+1}}(t)|}{P(t)} < \delta
\]

- Kreuzungen von MAs von unten nach oben (z.B. \( MA_{20} \) kreuzt \( MA_{50} \)).

#### 2.3.3. Erkennung von Spitzen

Spitzen werden identifiziert, wenn schnelle Kursanstiege stattfinden und bestimmte Kreuzungen der MAs beobachtet werden.

**Kriterien**:

- \( MA_9(t) \) kreuzt \( MA_{20}(t) \) von unten nach oben (Beginn der Spitze).
- Später kreuzt \( MA_{20}(t) \) \( MA_{50}(t) \) von oben nach unten (Ende der Spitze).

#### 2.3.4. Erkennung von Aufwärts- und Abwärtstrends

**Aufwärtstrend**:

- \( MA_9(t) > MA_{20}(t) > MA_{50}(t) > MA_{100}(t) \)

**Abwärtstrend**:

- \( MA_9(t) < MA_{20}(t) < MA_{50}(t) \)

#### 2.3.5. Generierung von Kauf- und Verkaufssignalen

Kauf- und Verkaufssignale basieren auf den Kreuzungen der MAs:

- **Kaufsignal**: \( MA_{20}(t) \) kreuzt \( MA_{50}(t) \) von unten nach oben.
- **Verkaufssignal**: \( MA_{20}(t) \) kreuzt \( MA_{50}(t) \) von oben nach unten.

### 2.4. Zeitreihenanalyse mit ARIMA

Zur Prognose zukünftiger Kursbewegungen und zur Bestätigung von Handelssignalen verwenden wir das ARIMA-Modell (Autoregressive Integrated Moving Average).

#### 2.4.1. Stationaritätsprüfung

Vor der Modellanpassung überprüfen wir die Zeitreihe auf Stationarität mittels des Augmented Dickey-Fuller-Tests:

- Nullhypothese \( H_0 \): Die Zeitreihe ist nicht stationär.
- Wenn der p-Wert \( p \leq 0{,}05 \), wird \( H_0 \) verworfen.

#### 2.4.2. Modellanpassung

Das ARIMA-Modell wird angepasst mit den Parametern \( (p, d, q) \):

- \( p \): Ordnung des autoregressiven Teils.
- \( d \): Anzahl der Differenzierungen (1, wenn nicht stationär).
- \( q \): Ordnung des gleitenden Durchschnittsteils.

Wir verwenden:

\[
\text{ARIMA}(1, d, 1)
\]

#### 2.4.3. Prognose und Signalbestätigung

Die Prognose für \( s \) Schritte in die Zukunft wird berechnet. Das Signal wird bestätigt, wenn die Prognose mit dem ursprünglichen Signal übereinstimmt:

- Bei einem **Kaufsignal** muss die Prognose einen Aufwärtstrend anzeigen.
- Bei einem **Verkaufssignal** muss die Prognose einen Abwärtstrend anzeigen.

### 2.5. Gewinnberechnung

Wir simulieren Trades basierend auf den erkannten Signalen und berechnen den potenziellen Gewinn.

#### 2.5.1. Handelsregeln

- **Kauf**:

  - Investition des gesamten verfügbaren Kapitals.
  - Berechnung der BTC-Menge:

    \[
    \text{BTC}_{\text{holding}} = \frac{\text{Balance}}{P(t)}
    \]

  - Aktualisierung des Kontostands:

    \[
    \text{Balance} = 0
    \]

- **Verkauf**:

  - Verkauf des gesamten BTC-Bestands.
  - Aktualisierung des Kontostands:

    \[
    \text{Balance} = \text{BTC}_{\text{holding}} \times P(t)
    \]

  - Berechnung des Gewinns:

    \[
    \text{Gewinn} = \text{Balance} - \text{Investitionsbetrag}
    \]

- Am Ende des Analysezeitraums werden alle verbleibenden BTC verkauft.

#### 2.5.2. Gesamtergebnis

Der Gesamtgewinn wird berechnet als:

\[
\text{Gesamtgewinn} = \text{Endbalance} - \text{Investitionsbetrag}
\]

---

## 3. Ergebnisse

Wir haben das Analyse-Tool auf historische BTC-Kursdaten angewendet und die oben beschriebenen Methoden implementiert.

- **Erkannte Muster**:

  - Mehrere Bodenzonen und Spitzen wurden identifiziert.
  - Aufwärts- und Abwärtstrends wurden erfolgreich erkannt.

- **Handelssignale**:

  - Kauf- und Verkaufssignale wurden generiert basierend auf den Kreuzungen der MAs.
  - Die Verwendung des ARIMA-Modells half bei der Bestätigung dieser Signale.

- **Gewinnsimulation**:

  - Durch die Simulation der Trades basierend auf den Signalen wurde ein potenzieller Gesamtgewinn berechnet.
  - Die Ergebnisse variieren je nach gewähltem Investitionsbetrag und Zeitrahmen.

- **Visualisierung**:

  - Interaktive Diagramme wurden erstellt, die Kursdaten, MAs, Handelssignale und Prognosen darstellen.

---

## 4. Diskussion

Die Verwendung von gleitenden Durchschnitten zur Mustererkennung in Finanzzeitreihen ist eine etablierte Methode. Die Kombination mit statistischen Modellen wie ARIMA bietet zusätzliche Einblicke und erhöht die Zuverlässigkeit der Signale.

- **Vorteile**:

  - Einfache Implementierung und Interpretation.
  - Möglichkeit zur Anpassung der Parameter an spezifische Bedürfnisse.
  - Interaktive Funktionen ermöglichen eine dynamische Analyse.

- **Herausforderungen**:

  - Finanzmärkte sind volatil, und vergangene Muster garantieren keine zukünftigen Ergebnisse.
  - Die Wahl der Parameter (z.B. Zeitfenster der MAs, Schwellenwerte) beeinflusst die Ergebnisse erheblich.
  - ARIMA-Modelle setzen Stationarität voraus, was bei Finanzzeitreihen nicht immer gegeben ist.

- **Verbesserungspotenzial**:

  - Integration weiterer Indikatoren und technischer Analysen.
  - Verwendung von Machine-Learning-Algorithmen zur Mustererkennung.
  - Implementierung von Risikomanagement-Strategien (z.B. Stop-Loss, Take-Profit).

---

## 5. Schlussfolgerungen und Ausblick

Wir haben ein interaktives Analyse-Tool entwickelt, das gleitende Durchschnitte und statistische Modelle nutzt, um Muster in BTC-Kursdaten zu erkennen und potenzielle Handelsgewinne zu simulieren. Die Ergebnisse zeigen, dass diese Methoden nützliche Einblicke bieten können. Für zukünftige Arbeiten planen wir die Erweiterung des Tools um Machine-Learning-Techniken und fortgeschrittene Zeitreihenmodelle, um die Genauigkeit und Aussagekraft der Analysen zu erhöhen.

---

## 6. Literaturverzeichnis

1. Brock, W., Lakonishok, J., & LeBaron, B. (1992). *Simple Technical Trading Rules and the Stochastic Properties of Stock Returns*. The Journal of Finance, 47(5), 1731-1764.
2. Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. San Francisco: Holden-Day.
3. Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*. OTexts.

---

**Anmerkung**: Alle verwendeten Daten und Ergebnisse sind exemplarisch und dienen der Illustration der Methoden. Dieses Papier ist ein Bestandteil des Projekts *MarketWave Analyzer* und soll die wissenschaftlichen Grundlagen und Vorgehensweisen dokumentieren.