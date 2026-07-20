# Notatka do projektu — funkcje i przepływ danych

## Cel projektu

Projekt przewiduje `Machine failure`:

- `0` — brak awarii,
- `1` — awaria.

Główna zasada: `record_index` służy do połączenia predykcji z oryginalnym rekordem, ale nie jest cechą modelu. Flagi awarii (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) również nie są używane jako cechy, ponieważ prowadziłoby to do wycieku informacji o targetcie.

Wspólna reguła decyzji w ewaluacji to:

```python
y_pred = (y_proba >= threshold).astype(int)
```

Czyli rekord z prawdopodobieństwem dokładnie równym progowi jest oznaczany jako awaria.

---

## 1. `python/cleaning.py`

### `prepare_dataset(input_path, output_path=None, correct_inconsistent_labels=True) -> pd.DataFrame`

**Wejście:**

- `input_path` — ścieżka do surowego pliku CSV,
- `output_path` — opcjonalna ścieżka zapisu oczyszczonego CSV,
- `correct_inconsistent_labels` — czy poprawić 9 rekordów mających `Machine failure = 1`, ale wszystkie flagi awarii równe 0.

**Co robi:**

1. Wczytuje surowe dane.
2. Sprawdza, czy istnieją wymagane kolumny.
3. Tworzy cechę `Temperature difference`:
   `Process temperature - Air temperature`.
4. Tworzy cechę `Power [W]`:
   `Rotational speed * Torque * 2*pi/60`.
5. Opcjonalnie zmienia niespójne etykiety awarii z `1` na `0`.
6. Usuwa identyfikatory, temperaturę powietrza i flagi awarii.
7. Koduje `Type` przez one-hot encoding (`Type_L`, `Type_M`; typ H jest kategorią bazową).
8. Sprawdza oczekiwane kolumny i braki danych.
9. Opcjonalnie zapisuje wynik.

**Wyjście:** przygotowany `DataFrame` zawierający cechy i `Machine failure`.

### `main() -> None`

Punkt wejścia dla polecenia `python python/cleaning.py`. Wywołuje `prepare_dataset()` na domyślnych ścieżkach i wypisuje liczbę zapisanych rekordów.

---

## 2. `python/data_preparation.py`

### `prepare_data(input_path, output_dir) -> None`

**Wejście:** oczyszczony CSV oraz katalog wyjściowy.

**Co robi:**

1. Wczytuje dane i sprawdza wymagane kolumny.
2. Sprawdza braki danych, duplikaty i wartości targetu (`0`/`1`).
3. Oddziela cechy `X` od targetu `y`.
4. Wykonuje stały podział stratyfikowany: 80% train, 20% test, `random_state=42`.
5. Dopasowuje `StandardScaler` wyłącznie do numerycznych cech treningowych.
6. Skaluje train i test tym samym scalerem.
7. Sprawdza zgodność indeksów oraz upewnia się, że target i `record_index` nie są cechami.
8. Zapisuje sześć plików CSV z indeksem nazwanym `record_index`.

**Wyjście:** brak wartości zwracanej; powstają:
`X_train_raw.csv`, `X_test_raw.csv`, `X_train_scaled.csv`, `X_test_scaled.csv`, `y_train.csv`, `y_test.csv`.

---

## 3. `python/features.py`

### `add_osf_criterion(X) -> pd.DataFrame`

**Wejście:** `DataFrame` zawierający `Tool wear [min]` i `Torque [Nm]`.

**Co robi:** tworzy kopię danych i dodaje cechę:

```text
OSF criterion = Tool wear [min] * Torque [Nm]
```

**Wyjście:** nowy `DataFrame`; oryginalny nie jest modyfikowany. Cecha korzysta z parametrów dostępnych przed awarią, ale jej zaprojektowanie było zainspirowane analizą etykiet OSF, co trzeba opisać w raporcie.

---

## 4. `python/evaluation.py`

### `prepare_fold_data(X_train_fold, X_validation_fold) -> (DataFrame, DataFrame)`

Skaluje numeryczne cechy w jednym foldzie walidacji krzyżowej.

- **Wejście:** dane treningowe i walidacyjne bieżącego foldu.
- **Działanie:** dopasowuje `StandardScaler` tylko do `X_train_fold`, a następnie używa go do train i validation.
- **Wyjście:** przeskalowane kopie obu zbiorów.
- **Dlaczego:** zapobiega wyciekowi informacji ze zbioru walidacyjnego do skalera.
- `Type_L` i `Type_M` nie są skalowane.

### `get_metrics(y_true, y_proba, threshold=0.5) -> dict`

Z prawdopodobieństw tworzy klasy według `>= threshold`, a następnie oblicza:
`accuracy`, `precision`, `recall`, `f1`, `roc_auc` i `pr_auc`.

- **Wejście:** prawdziwe etykiety, prawdopodobieństwa klasy 1 i próg.
- **Wyjście:** słownik metryk.
- `zero_division=0` zapobiega błędowi, gdy model nie przewidzi żadnej klasy pozytywnej.

### `cross_validate(model, X, y, cv, threshold=0.5) -> (DataFrame, Series)`

Wykonuje pełną walidację krzyżową.

1. Pobiera indeksy train/validation z `cv`.
2. Skaluje dane przez `prepare_fold_data`.
3. Klonuje model, aby każdy fold miał niezależny model.
4. Trenuje model na train foldzie.
5. Oblicza prawdopodobieństwa na validation foldzie.
6. Zapisuje metryki foldu i predykcje out-of-fold.

**Wyjście:**

- `DataFrame` z metrykami dla każdego foldu,
- `Series` `y_proba_oof`, wyrównana do indeksu `X`.

Model jest trenowany raz na fold. Z tych samych prawdopodobieństw można potem testować różne progi bez ponownego trenowania.

### `summarize_fold_metrics(fold_metrics_df) -> dict`

Oblicza średnią i odchylenie standardowe każdej metryki w foldach. Zwraca klucze w rodzaju `f1_mean` i `f1_std`.

### `evaluate_thresholds(y_true, y_proba, thresholds) -> pd.DataFrame`

Porównuje wiele progów decyzyjnych na tych samych prawdopodobieństwach. Dla każdego progu zapisuje precision, recall, F1, liczbę FP, liczbę FN oraz `ties`, czyli liczbę rekordów z prawdopodobieństwem dokładnie równym progowi.

### `get_calibration_data(y_true, y_proba, n_bins=10) -> (DataFrame, float)`

Przygotowuje dane do diagramu kalibracji i oblicza Brier score.

- Dzieli prawdopodobieństwa na 10 równych przedziałów (`strategy="uniform"`).
- Dla niepustych binów zapisuje średnie przewidywane prawdopodobieństwo, rzeczywisty odsetek awarii i liczność binu.
- **Wyjście:** tabela kalibracji oraz Brier score.

### `get_threshold_ties(y_proba, threshold=0.5) -> pd.Series`

Zwraca rekordy, których prawdopodobieństwo dokładnie równa się progowi. Pomaga jawnie sprawdzić przypadki na granicy decyzji.

---

## 5. `python/visualization.py`

Wszystkie funkcje wizualizacyjne zwracają obiekt `matplotlib.figure.Figure`. Nie zapisują pliku samodzielnie — zapisuje go skrypt wywołujący przez `fig.savefig(...)`.

### `plot_confusion_matrix_grid(matrices) -> Figure`

Tworzy siatkę macierzy pomyłek. Słownik wejściowy ma strukturę `wariant_cech -> model -> macierz`.

### `plot_probability_distribution(y_true, y_proba, thresholds=None, title=...) -> Figure`

Rysuje histogram prawdopodobieństw osobno dla klasy 0 i 1. Opcjonalnie zaznacza pionowymi liniami progi decyzyjne. Oś Y jest logarytmiczna, aby widzieć również rzadkie obserwacje.

### `plot_calibration_curve(calibration_tables, title=...) -> Figure`

Rysuje reliability diagram dla jednego lub kilku źródeł prawdopodobieństw. Rozmiar punktu zależy od liczby rekordów w binie, a legenda zawiera Brier score.

### `plot_drift_report(drift_report) -> Figure`

Tworzy poziomy wykres PSI/proportion difference i koloruje wartości PSI według progów: `<0.1` dobrze, `0.1–0.25` ostrzeżenie, `>0.25` krytyczny dryf.

### `plot_threshold_analysis(threshold_results, feature_variant, model_names) -> Figure`

Tworzy dwa wykresy: metryki względem progu oraz liczby FP/FN względem progu. Oczekuje tabeli z `evaluate_thresholds`, rozszerzonej o kolumny `feature_variant` i `model`.

---

## 6. Funkcje z notebooków

Notebooki zawierają również funkcje lokalne. Część z nich jest historyczną kopią funkcji przeniesionych później do katalogu `python/`. W nowych analizach należy preferować moduły z `python/`, ponieważ są współdzielone i łatwiejsze do testowania.

### `notebooks/01_eda.ipynb` oraz `notebooks/01_eda — kopia.ipynb`

#### `plot_box_hist(df, flag_col, value_col, palette="Set1")`

Tworzy wykres eksploracyjny łączący rozkład wartości z podziałem według flagi awarii. `df` to tabela danych, `flag_col` określa kolumnę grupującą, `value_col` określa badaną zmienną, a `palette` — paletę kolorów. Funkcja służy do EDA, a nie do trenowania modelu.

### `notebooks/04_modeling.ipynb`

#### `predict_with_threshold(model, X, threshold=0.5)`

Wykonuje jedno wspólne obliczenie `predict_proba`, a następnie zwraca prawdopodobieństwa klasy 1 i klasy według reguły `>= threshold`. Zapobiega rozjechaniu się tabel metryk i analiz błędów.

#### `get_metrics(name, y_true, y_pred, y_proba, threshold=0.5)`

Tworzy słownik metryk dla jednego modelu: nazwę modelu, próg, accuracy, precision, recall, F1, ROC-AUC i PR-AUC. Przyjmuje już obliczone klasy i prawdopodobieństwa.

#### `evaluate_model(name, y_true, y_pred, y_proba, threshold=0.5)`

Wywołuje `get_metrics`, drukuje metryki, macierz pomyłek i classification report. Jest funkcją prezentacyjną do notebooka; nie zwraca osobnego wyniku.

#### `get_false_negatives(debug_df, model_name)`

Filtruje tabelę diagnostyczną i zwraca rekordy, dla których prawdziwa klasa to `1`, a wskazany model przewidział `0`.

#### `add_failure_mechanism_flags(df)`

Dodaje do tabeli diagnostyczne kolumny opisujące podobieństwo do mechanizmów awarii: `OSF_distance`, `OSF_like`, `HDF_like`, `PWF_like` i `TWF_like`. Są to reguły do interpretacji błędów, nie cechy modelu. Opierają się na progach zaobserwowanych w tym zbiorze AI4I.

### `notebooks/05_feature_experiment.ipynb`

#### `prepare_fold_data(X_train_fold, X_validation_fold)`

Notebookowa, wcześniejsza wersja funkcji z `python/evaluation.py`. Skaluje cechy numeryczne tylko na train foldzie i stosuje scaler do validation folda.

#### `get_metrics(y_true, y_proba)`

Notebookowa wersja liczenia metryk. Używa stałego progu `DECISION_THRESHOLD` i zwraca accuracy, precision, recall, F1, ROC-AUC oraz PR-AUC. Jej bardziej elastycznym odpowiednikiem jest `evaluation.get_metrics`.

#### `evaluate_model_with_cv(model, X, y, cv)`

Trenuje model osobno na każdym foldzie, skaluje dane foldowo i zwraca metryki dla foldów. Była używana do porównywania wariantów cech i modeli.

#### `get_oof_predictions(model, X, y, cv)`

Wykonuje CV i zwraca połączone OOF etykiety prawdziwe oraz przewidziane. Służy do budowania macierzy pomyłek z predykcji, w której każdy rekord jest oceniony przez model, który go nie trenował.

#### `get_oof_probabilities(model, X, y, cv)`

Wykonuje foldową normalizację, trening i zapisuje prawdopodobieństwo klasy 1 w odpowiednim miejscu serii OOF. Jest poprzednikiem połączonej funkcji `evaluation.cross_validate`.

#### `evaluate_thresholds(y_true, y_proba, thresholds)`

Notebookowa wersja testowania progów. Dla każdego progu liczy precision, recall, F1, FP i FN. Nowsza wersja w `python/evaluation.py` dodatkowo raportuje remisy (`ties`).

#### `get_error_records(feature_variant, model_name, threshold=0.5)`

Buduje tabelę błędów na podstawie zapisanych OOF prawdopodobieństw. Dodaje `y_pred` i `error_type`, a następnie zwraca tylko rekordy `FP` i `FN` dla wybranego wariantu cech, modelu i progu.

#### `calculate_alert_metrics(y_true, y_pred)`

Rozpakowuje macierz pomyłek do `TN`, `FP`, `FN`, `TP` i zwraca precision, recall, F1 oraz liczności wszystkich czterech kategorii.

### `notebooks/experiment.ipynb`

#### `get_best_model(results)`

Wybiera najlepszy model z tabeli wyników na podstawie ustalonego kryterium notebooka. Jest pomocniczą funkcją eksploracyjną; ostateczny wybór modelu został później przeniesiony do `python/final_model.py`.

---

## 7. `python/drift_monitoring.py`

### `population_stability_index(reference, current, n_bins=10) -> float`

Oblicza PSI dla nowego zbioru `current` względem `reference`.

- Przedziały są wyznaczane na podstawie decyli zbioru referencyjnego.
- Małe `PSI_EPSILON` zapobiega logarytmowi z zera.
- Interpretacja: `<0.1` brak istotnej zmiany, `0.1–0.25` zmiana umiarkowana, `>0.25` zmiana znacząca.

### `compute_drift_report(reference, current, features) -> pd.DataFrame`

Dla każdej cechy oblicza test KS oraz właściwą miarę:

- numeryczne — PSI,
- `Type_L`, `Type_M` — różnicę proporcji.

Zwraca tabelę z `metric`, `value`, `ks_statistic` i `ks_pvalue`.

---

## 8. Skrypty uruchomieniowe i ich `main()`

### `python/final_model.py`

- `load_reduced_data()` — wczytuje train/test, dodaje OSF criterion i usuwa redundantne `Torque [Nm]` oraz `Process temperature [K]`.
- `main()` — wykonuje OOF cross-validation, testuje progi dla kosztów FN:FP, zapisuje `final_threshold_costs.csv`, trenuje finalny Gradient Boosting na pełnym train, sprawdza model na X_test i zapisuje `final_model_summary.csv` oraz `models/final_model.joblib`.

### `python/sensitivity_analysis.py`

- `main()` — porównuje wariant z korektą 330 awarii i bez korekty 339 awarii. Dla Random Forest i Gradient Boosting wykonuje 5-fold CV i zapisuje `results/sensitivity_target_correction.csv`.

### `python/check_drift.py`

- `main()` — używa `X_train_raw` jako referencji i `X_test_raw` jako bieżącego zbioru kontrolnego, dodaje OSF criterion, wywołuje `compute_drift_report`, zapisuje CSV i wykres. To sanity check, nie pomiar realnego dryfu produkcyjnego.

### `python/plot_calibration.py`

- `main()` — wczytuje artefakt finalnego modelu, oblicza prawdopodobieństwa OOF dla train i predykcje X_test, wywołuje `get_calibration_data`, zapisuje tabelę oraz wykres kalibracji.

### `python/plot_final_probabilities.py`

- `main()` — wczytuje finalny artefakt, skaluje X_test zgodnie z zapisanym scalerem, oblicza prawdopodobieństwa i zapisuje histogram rozkładu prawdopodobieństw z progami 0.30 i 0.50.

---

## Najważniejszy przepływ danych

```text
surowe CSV
  -> cleaning.prepare_dataset
  -> produkcja_clean.csv
  -> data_preparation.prepare_data
  -> train/test raw + scaled + target
  -> add_osf_criterion
  -> cross_validate / final model
  -> metryki, progi, wykresy i artefakt joblib
```

## Ważne rozróżnienie wyników

- Wyniki OOF z `cross_validate` służą do porównania i wyboru progu na train.
- `X_test` jest obecnie development checkiem, ponieważ był używany podczas eksploracji.
- Prawdziwa ocena wdrożeniowa wymaga przyszłych, nietkniętych danych.
- Dobór ostatecznego progu wymaga rzeczywistych kosztów biznesowych FN i FP.
