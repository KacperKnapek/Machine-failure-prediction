# PROJECT_CONTEXT.md

Stały kontekst projektu predykcji awarii maszyn. Przed pracą z repozytorium należy przeczytać ten plik, a następnie notebooki.

**Ostatni przegląd:** 2026-07-13  
**Przejrzany commit `main`:** `bfd0e7ab22463a2071ec59cf3fae05154b9b962c`

---

## 0. Aktualny status

Dzisiaj wykonano wartościowy postęp:

- zakończono analizę false positives dla Random Forest i Gradient Boosting,
- dodano trwałe eksporty do `results/`,
- dodano `X_train_raw.csv` i `X_test_raw.csv`,
- podjęto próbę zachowania indeksu jako `record_index`,
- uporządkowano kod `04_modeling.ipynb`.

Po ostatniej strukturyzacji wystąpiła jednak krytyczna niespójność:

```text
Kod notebooków, zapisane CSV, outputy notebooka i pliki results/
nie pochodzą obecnie z jednego spójnego uruchomienia pipeline.
```

Przed dalszym modelowaniem trzeba wykonać naprawy z sekcji 9.

---

## 1. Cel i najważniejsze zasady

Cel:

```text
Machine failure: 0 = brak awarii, 1 = awaria
```

Nie używać jako cech modelu:

```text
TWF, HDF, PWF, OSF, RNF
UDI, Product ID
record_index
```

Flagi typów awarii służą do EDA i interpretacji błędów. Ich użycie jako X spowodowałoby data leakage. `record_index` ma być wyłącznie metadanym łączącym predykcję z oryginalnym rekordem.

---

## 2. Struktura repozytorium

```text
PROJECT_CONTEXT.md
README.md
TO-DO - projekt.md

data/raw/produkcja.csv

data/processed/produkcja_clean.csv
data/processed/X_train_raw.csv
data/processed/X_test_raw.csv
data/processed/X_train_scaled.csv
data/processed/X_test_scaled.csv
data/processed/y_train.csv
data/processed/y_test.csv

notebooks/01_eda.ipynb
notebooks/02_data_cleaning.ipynb
notebooks/03_data_preparation.ipynb
notebooks/04_modeling.ipynb

python/cleaning.py

results/model_results_baseline.csv
results/false_negatives_baseline.csv
results/false_positives_baseline.csv
```

Do utworzenia lub rozbudowy:

```text
reports/
requirements.txt
pełny README.md
```

---

## 3. Dane i decyzje projektowe

Dane surowe:

```text
10000 rekordów
brak braków i duplikatów
UDI i Product ID są unikalne
```

Rozkład typu:

```text
L = 6000
M = 2997
H = 1003
```

Przed korektą było 339 awarii. Dziewięć rekordów miało `Machine failure = 1`, lecz wszystkie flagi typu awarii równe 0. Ustawiono je na 0.

Po korekcie:

```text
Machine failure = 1: 330
Machine failure = 0: 9670
```

To jest założenie projektowe, nie bezsporna prawda. Później należy wykonać sensitivity analysis dla wariantu 330 i 339 awarii.

Cechy inżynierskie:

```text
Power [W] = rpm * Torque * 2*pi/60
Temperature difference = Process temperature - Air temperature
```

Charakterystyczne obszary awarii:

```text
TWF: Tool wear [min] = 198–253
HDF: Temperature difference = 7.6–8.6 i rpm = 1212–1379
PWF: Power = 1148.44–3477.24 albo 9004.43–10469.92
OSF: Tool wear * Torque, obserwowany próg około 11003.2
```

---

## 4. Przygotowanie danych

Założony podział:

```text
train = 8000
development/test = 2000
random_state = 42
stratify = y
```

Rozkład klas:

```text
train: 7736 / 264
test: 1934 / 66
```

Skalowane cechy:

```text
Process temperature [K]
Temperature difference
Rotational speed [rpm]
Torque [Nm]
Power [W]
Tool wear [min]
```

Nie skalować `Type_L` i `Type_M`. `StandardScaler` dopasowywać tylko na train.

Pliki z `record_index` należy wczytywać tak:

```python
X_train = pd.read_csv("../data/processed/X_train_scaled.csv", index_col="record_index")
X_test = pd.read_csv("../data/processed/X_test_scaled.csv", index_col="record_index")
y_train = pd.read_csv("../data/processed/y_train.csv", index_col="record_index")["Machine failure"]
y_test = pd.read_csv("../data/processed/y_test.csv", index_col="record_index")["Machine failure"]

assert X_train.index.equals(y_train.index)
assert X_test.index.equals(y_test.index)
assert "record_index" not in X_train.columns
assert "record_index" not in X_test.columns
```

---

## 5. Modele bazowe — historyczny spójny wynik

```text
model                 accuracy  precision  recall   f1      roc_auc  pr_auc
Dummy                 0.9670    0.0000     0.0000   0.0000  0.5000   0.0330
Logistic Regression   0.8475    0.1595     0.8485   0.2686  0.9202   0.4192
Random Forest         0.9900    0.8594     0.8333   0.8462  0.9910   0.8930
Gradient Boosting     0.9895    0.8814     0.7879   0.8320  0.9943   0.9145
```

Random Forest dawał najlepszy kompromis precision/recall/F1. Gradient Boosting miał najlepszy ROC-AUC i PR-AUC. Logistic Regression miała wysoki recall, ale 295 false positives.

### Niespójność wyników po dzisiejszych zmianach

Aktualny `results/model_results_baseline.csv` podaje dla Random Forest:

```text
accuracy = 0.9895
precision = 0.846154
recall = 0.833333
f1 = 0.839695
```

To odpowiada 10 false positives. Tymczasem `results/false_positives_baseline.csv` zawiera 9 rekordów Random Forest.

Wniosek:

```text
Pliki results/ należy wygenerować ponownie w jednym czystym uruchomieniu.
```

---

## 6. False negatives

```text
Logistic Regression: 10
Random Forest: 11
Gradient Boosting: 14
```

Przeoczone przez wszystkie modele:

```text
8199, 8609, 4034, 9018, 7510
```

Grupy mechanizmów:

```text
PWF boundary: 442
HDF boundary: 3760, 3787, 3829, 4151, 4833, 4475
OSF bez jawnej interakcji: 1085, 2494, 9653, 9663, 8026
TWF wysokie rpm i wear: 2941, 9758
TWF wysokie wear: 7510, 9018, 4034, 8609, 8199
```

Hipoteza do kontrolowanego testu:

```text
Kryterium_OSF = Tool wear [min] * Torque [Nm]
```

---

## 7. False positives — analiza z 2026-07-13

Eksport:

```text
results/false_positives_baseline.csv
```

Aktualny plik zawiera:

```text
Random Forest: 9
Gradient Boosting: 7
wspólne: 3764, 7677, 4862
```

Zakresy prawdopodobieństw:

```text
RF: 0.530–0.750
GB: 0.543–0.855
margin = predicted_probability - 0.5
```

Definicje podobieństwa:

```text
OSF_like: Tool wear * Torque >= 11003.2
HDF_like: temp diff w [7.6, 8.6] i rpm w [1212, 1379]
PWF_like: moc w jednym z dwóch przedziałów PWF
TWF_like: Tool wear w [198, 253]
```

Liczności:

```text
RF: HDF_like 4, TWF_like 5, OSF_like 1, PWF_like 0
GB: HDF_like 2, TWF_like 5, OSF_like 2, PWF_like 0
```

Każdy zapisany false positive przypomina co najmniej jeden obszar awarii. Mały margin sugeruje przypadek bliski progowi; duży margin sugeruje silny wzorzec awarii albo możliwy problem etykiety.

---

## 8. Ocena dzisiejszych zmian

### Mocne strony

- analiza błędów została połączona z mechaniką awarii,
- dodano `predicted_probability`, `margin`, `OSF_distance` i flagi podobieństwa,
- wyniki przeniesiono z outputów notebooka do `results/`,
- rozdzielono surowe i skalowane cechy,
- zachowanie indeksu jest właściwym kierunkiem,
- zauważono, że dotychczasowy test stał się development setem.

### Słabe strony

- `04_modeling.ipynb` nie przechodzi obecnie od początku do końca,
- kod `03_data_preparation.ipynb` nie odtwarza aktualnych CSV,
- `record_index` może zostać użyty jako cecha,
- target może zostać wczytany jako dwukolumnowy DataFrame,
- zapisane outputy są starsze niż aktualne CSV,
- pliki wynikowe są niespójne dla Random Forest.

```text
Postęp analityczny: bardzo dobry.
Organizacja repozytorium: dobry kierunek.
Aktualna reprodukowalność: niewystarczająca do dalszych eksperymentów.
```

---

## 9. Błędy do naprawy teraz

### Funkcje metryk

W `evaluate_model` i `get_metrics` najpierw utworzyć `y_proba`, potem `y_pred`:

```python
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= threshold).astype(int)
```

Aktualne `y.proba` jest błędem.

### Zapis danych w `03_data_preparation.ipynb`

Notebook powinien zapisywać raw, scaled i y z indeksem:

```python
X_train_raw.to_csv("../data/processed/X_train_raw.csv", index=True, index_label="record_index")
X_test_raw.to_csv("../data/processed/X_test_raw.csv", index=True, index_label="record_index")
X_train.to_csv("../data/processed/X_train_scaled.csv", index=True, index_label="record_index")
X_test.to_csv("../data/processed/X_test_scaled.csv", index=True, index_label="record_index")
y_train.to_frame().to_csv("../data/processed/y_train.csv", index=True, index_label="record_index")
y_test.to_frame().to_csv("../data/processed/y_test.csv", index=True, index_label="record_index")
```

Aktualny kod nie zapisuje raw i używa `index=False`, mimo że pliki w repo zawierają `record_index`.

### Kolejność naprawy

1. Poprawić `03_data_preparation.ipynb`.
2. Uruchomić go od początku.
3. Poprawić wczytywanie przez `index_col="record_index"` w `04_modeling.ipynb`.
4. Poprawić funkcje metryk.
5. Restart Kernel i Run All.
6. Wygenerować ponownie wszystkie pliki `results/` w jednym przebiegu.
7. Sprawdzić:

```text
TP + FN = 66
TN + FP = 1934
accuracy = (TP + TN) / 2000
precision = TP / (TP + FP)
recall = TP / (TP + FN)
```

---

## 10. Metodologia dalszej pracy

Obecny zbiór 2000 rekordów wykorzystano do porównania modeli, analizy błędów, projektowania cech i planowania progów. Należy go traktować jako development set.

Preferowany schemat:

```text
train + StratifiedKFold cross-validation
final untouched test
```

Następny eksperyment:

```text
A: obecne 8 cech
B: obecne 8 cech + Kryterium_OSF
```

Porównywać precision, recall, F1, PR-AUC, odchylenie standardowe i false negatives. Threshold tuning wykonać dopiero po wyborze modelu i zestawu cech.

---

## 11. Zasady dla kolejnych asystentów

```text
Najpierw naprawić reprodukowalność.
Nie używać flag awarii, identyfikatorów ani record_index jako X.
Nie oceniać modelu wyłącznie przez accuracy.
Pilnować recall, precision, F1 i PR-AUC.
Oddzielać fakty, interpretacje i hipotezy.
Nie stroić na finalnym teście.
Testować pojedyncze zmiany, a dopiero potem je łączyć.
```