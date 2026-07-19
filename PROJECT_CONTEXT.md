# PROJECT_CONTEXT.md

## Update 2026-07-19 - reduced feature set and temperature analysis (notebook 05)

Controlled experiment removing both near-zero features flagged by the
importance analysis (`Torque [Nm]`, `Process temperature [K]`):

```text
CV (GB, 5-fold):  9 features F1 0.8954, PR-AUC 0.9258 | 7 features F1 0.8900, PR-AUC 0.9154 (within fold std)
X_test:           identical outcome for both sets: precision 0.9825, recall 0.8485, F1 0.9106, FP 1, FN 10
                  (PR-AUC 0.9368 for 7 features vs 0.9357 for 9)
```

The 7-feature reduced set is a candidate for the final model — no
measurable loss on the development check with fewer inputs to monitor.

Why absolute process temperature carries no signal (documented with class
distributions in the notebook): failures and non-failures have nearly the
same absolute process temperature (means 310.27 vs 310.00 K, correlation
with the target 0.033), while `Temperature difference` separates them
(9.38 vs 10.02 K, correlation -0.114; HDF band 7.6-8.6 K). In this data
the process temperature is ambient plus roughly 10 K, so the absolute
level tracks ambient drift; the condition signal is the gradient. The
conclusion is redundancy given `Temperature difference` in this dataset,
not general unimportance of temperature.

## Update 2026-07-19 - feature importance (notebook 05)

Notebook 05 gained a feature-importance section for the final Gradient
Boosting + OSF model: built-in impurity importances of the final model and
permutation importance computed per CV fold on the validation part only
(PR-AUC scoring, `n_repeats=10`, `random_state=42`).

```text
impurity:    TempDiff 0.3465, Power 0.2770, OSF 0.2437, rpm 0.0598, ToolWear 0.0359, Torque 0.0110, ProcessTemp 0.0078
permutation: Power 0.4252±0.0883, rpm 0.3245±0.0285, TempDiff 0.2911±0.0197, OSF 0.2267±0.0475, ToolWear 0.0637±0.0396, Torque 0.0037, ProcessTemp -0.0016
```

Conclusions:

- The OSF criterion carries real signal in both views and stays in the
  final feature set.
- Raw Torque contributes almost nothing (consistent with the
  remove-Torque experiment); Process temperature is a removal candidate
  for a controlled follow-up experiment.
- With correlated features (Power = rpm * torque * 2*pi/60) the ranking
  should be read as a group, not as exact per-feature values.

The notebook was re-executed end to end; the final X_test metrics
reproduced exactly (precision 0.9825, recall 0.8485, FP 1, FN 10).

## Update 2026-07-19 - task audit and remaining section 10 items

Audit of section 10: tasks 1, 3, 6, 7, 9, 10 were already done; task 11
(threshold tuning from explicit costs) is in progress in notebook 05.
The remaining tasks were completed today:

- **10.2** — `evaluation.get_threshold_ties` and a `ties` column in
  `evaluate_thresholds`; notebook 04 now explicitly prints records with
  probability equal to the threshold (Random Forest: record 5653).
- **10.4** — `false_positives_baseline.csv` regained the raw physical
  feature columns and is self-sufficient again.
- **10.5** — report copies of probabilities, margin and OSF_distance are
  rounded to 4 decimals; computations keep full precision.
- **10.8** — README gained a "Key results" section with the baseline table
  and the GB + OSF development result, including the dev-set caveat.
- **10.12** — sensitivity analysis of the 9-label correction
  (`python/sensitivity_analysis.py`, results in
  `results/sensitivity_target_correction.csv`). 5-fold CV of RF and GB on
  baseline + OSF features under both label variants:

  ```text
  GB corrected   (330): precision 0.9651±0.0468, recall 0.8409±0.0860, f1 0.8954, roc_auc 0.9909, pr_auc 0.9258
  GB uncorrected (339): precision 0.9574±0.0152, recall 0.8303±0.0237, f1 0.8893, roc_auc 0.9803, pr_auc 0.9083
  RF corrected   (330): precision 0.8833±0.0608, recall 0.8525±0.0935, f1 0.8622, roc_auc 0.9891, pr_auc 0.9117
  RF uncorrected (339): precision 0.9043±0.0248, recall 0.8339±0.0393, f1 0.8673, roc_auc 0.9757, pr_auc 0.8952
  ```

  Conclusion: keeping the original 339 labels slightly lowers ranking
  metrics (the 9 unexplained failures behave like label noise) but the
  differences stay within fold-to-fold variation and the model ranking is
  unchanged. The 330-label correction is not driving the project results.

Also fixed: notebook 04 `debug_df` now really skips the dummy model (the
old check compared against the wrong key), and `visualization.py`
annotates figures with `matplotlib.figure.Figure` instead of `plt.Figure`.
Notebook 04 was re-executed end to end; `model_results_baseline.csv` is
bit-identical and the FP/FN exports contain the same records and values.
`cleaning.prepare_dataset` accepts `correct_inconsistent_labels` and an
optional output path; default behaviour is unchanged (verified by
regenerating `produkcja_clean.csv` bit-identically).

## Update 2026-07-19 - housekeeping

- Model naming is unified: `Logistic Regression` everywhere (the `Log Reg` inconsistency from section 0 no longer exists).
- `PROJECT_CONTEXT.md` was removed from `.gitignore` (task 10.6 done); `context.md` (compact machine-readable summary) is now tracked.
- Reusable helpers were extracted from `notebooks/05_feature_experiment.ipynb` (task 10.9 and "consolidate OOF helpers" done):
  - `python/features.py` — `add_osf_criterion`,
  - `python/evaluation.py` — `prepare_fold_data`, `get_metrics`, `cross_validate`, `summarize_fold_metrics`, `evaluate_thresholds`,
  - `python/visualization.py` — `plot_confusion_matrix_grid`, `plot_threshold_analysis`.
- `cross_validate` consolidates the three notebook fold loops into one training pass per fold and returns both per-fold metrics and out-of-fold probabilities. It was verified to produce bit-identical results to the original notebook functions and to reproduce the documented GB+OSF fold 5 result (`[[1547, 0], [16, 37]]`).
- Notebook 05 keeps its own inline copies as a historical record; new notebooks should import from `python/` instead.

## Update 2026-07-19 - OSF feature experiment

The separate notebook `notebooks/05_feature_experiment.ipynb` now compares the baseline feature set with the baseline plus:

```python
OSF criterion = Tool wear [min] * Torque [Nm]
```

The experiment uses 5-fold `StratifiedKFold`, fold-local scaling, out-of-fold predictions, confusion matrices, threshold analysis, FP/FN inspection, cost scenarios, and fold-stability analysis. The shared decision rule remains `y_pred = (y_proba >= 0.5).astype(int)`.

The best current variant is `Gradient Boosting + OSF criterion` at threshold `0.5`. On `X_test` it achieved precision `0.9825`, recall `0.8485`, F1 `0.9106`, ROC-AUC `0.9953`, PR-AUC `0.9357`, 1 false positive, and 10 false negatives. The confusion matrix was `[[1933, 1], [10, 56]]`.

Compared with the previous Gradient Boosting baseline, false positives decreased from 7 to 1 and false negatives from 14 to 10. This is a strong development result, but `X_test` was already used during exploration and OSF design, so it is not a completely untouched final validation.

Fold 5 was harder for several models. For Gradient Boosting + OSF it had precision `1.0`, recall `0.6981`, and confusion matrix `[[1547, 0], [16, 37]]`. Similar recall degradation for Random Forest suggests a harder fold rather than a problem specific to OSF or one model.

Next: inspect the 10 test false negatives, choose a threshold using explicit FP/FN costs, consolidate the OOF helper functions, inspect feature importance, and validate on future or external data.

## Aktualizacja 2026-07-16

Stan po dzisiejszych zmianach:

- przygotowanie danych zostalo wyodrebnione do `python/data_preparation.py`,
- skrypt waliduje wymagane kolumny, braki danych, duplikaty, target i zgodnosc indeksow,
- skrypt wykonuje staly podzial 80/20, skaluje wylacznie kolumny numeryczne i zapisuje szesc artefaktow CSV,
- notebooki `02_data_cleaning.ipynb`, `03_data_preparation.ipynb` i `04_modeling.ipynb` zostaly dostosowane do tego kierunku,
- do `.gitignore` dodano `__pycache__/`.

Dzisiejsze zmiany obejmuja commity `956551a`, `c8d55bb` i `ae5cdef`. Aktualny HEAD `ae5cdefb3779bc63193f67ada5a3461d20b385df` jest zgodny z `origin/main`.


Stały kontekst projektu predykcji awarii maszyn. Przed pracą z repozytorium należy przeczytać ten plik, a następnie odpowiednie notebooki i pliki wynikowe.

**Ostatni przegląd:** 2026-07-16  
**Przejrzany commit `main`:** `ae5cdefb3779bc63193f67ada5a3461d20b385df`

---

## 0. Aktualny status

Po zmianach z 2026-07-14 pipeline bazowy jest ponownie odtwarzalny i znacznie lepiej uporządkowany.

Sprawdzone commity:

```text
548577ca27db9eba4e8181bb7a198ced73a0be03
Fix reproducible data preparation and baseline analysis

8b8d16b53aa6bcb679b10c81646f560a544d1f5d
structure notebooks

6b1bf4093e6ef4dee56c60777537d730d8b5f9e2
structure notebooks

7bbea413c9680707cb8edc9684b6bc64a6e8ddc4
refactor(modeling): simplify baseline evaluation and refresh error analysis
```

Najważniejsze rezultaty:

- `03_data_preparation.ipynb` zapisuje surowe i skalowane zbiory wraz z `record_index`,
- `04_modeling.ipynb` wczytuje `record_index` jako indeks, a nie jako cechę,
- target jest wczytywany jako jednowymiarowa seria,
- predykcje i prawdopodobieństwa są wyliczane wspólną funkcją,
- pliki w `results/` zostały wygenerowane ponownie,
- metryki i listy błędów są obecnie wewnętrznie spójne przy przyjętej regule progu `>= 0.5`,
- notebooki mają czytelniejszą strukturę i opis etapów analizy.

Aktualna ocena:

```text
Postęp analityczny: bardzo dobry.
Reprodukowalność pipeline: dobra dla obecnego eksperymentu bazowego.
Organizacja notebooków: wyraźnie poprawiona.
Gotowość do dalszych eksperymentów: tak, po zachowaniu ostrożności metodologicznej opisanej niżej.
```

Pozostałe problemy nie blokują pracy, ale powinny zostać uporządkowane:

1. Jawnie udokumentować zachowanie rekordu z prawdopodobieństwem dokładnie `0.5`.
2. Ujednolicić nazwę `Log Reg` / `Logistic Regression`.
3. Rozważyć ponowne dodanie surowych parametrów fizycznych do eksportu false positives.
4. Usunąć `PROJECT_CONTEXT.md` z `.gitignore`, ponieważ plik jest celowo wersjonowany.
5. Nie traktować obecnego zbioru 2000 rekordów jako nietkniętego finalnego testu.

---

## 1. Cel i najważniejsze zasady

Cel modelowania:

```text
Machine failure: 0 = brak awarii, 1 = awaria
```

Projekt jest obecnie klasyfikacją binarną, a nie wieloetykietową predykcją typów awarii.

Nie używać jako cech modelu:

```text
TWF
HDF
PWF
OSF
RNF
UDI
Product ID
record_index
```

Powody:

- `TWF`, `HDF`, `PWF`, `OSF`, `RNF` są bardzo blisko związane z targetem i spowodowałyby data leakage,
- `UDI` jest numerem rekordu,
- `Product ID` jest unikalny i koduje `Type`,
- `record_index` służy wyłącznie do połączenia predykcji z oryginalnym rekordem.

Przy niezbalansowanym targecie nie oceniać modelu wyłącznie przez accuracy.

Najważniejsze metryki:

```text
precision klasy 1
recall klasy 1
F1 klasy 1
PR-AUC
ROC-AUC
confusion matrix
liczba false negatives
liczba false positives
```

---

## 2. Struktura repozytorium

```text
PROJECT_CONTEXT.md
README.md
TO-DO - projekt.md
.gitignore

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
python/data_preparation.py

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
brak braków danych
brak duplikatów
UDI i Product ID są unikalne
```

Rozkład typu maszyny:

```text
L = 6000
M = 2997
H = 1003
```

Przed korektą targetu:

```text
Machine failure = 1: 339
Machine failure = 0: 9661
```

Dziewięć rekordów miało `Machine failure = 1`, ale wszystkie flagi typu awarii były równe 0. W projekcie ustawiono je na 0.

Po korekcie:

```text
Machine failure = 1: 330
Machine failure = 0: 9670
```

To jest założenie projektowe, a nie bezsporna prawda. W późniejszym etapie warto wykonać sensitivity analysis dla wariantów z 330 i 339 awariami.

Cechy inżynierskie:

```text
Power [W] = Rotational speed [rpm] * Torque [Nm] * 2*pi/60
Temperature difference = Process temperature [K] - Air temperature [K]
```

Charakterystyczne obszary awarii z EDA:

```text
TWF: Tool wear [min] = 198–253
HDF: Temperature difference = 7.6–8.6 i rpm = 1212–1379
PWF: Power [W] = 1148.44–3477.24 albo 9004.43–10469.92
OSF: Tool wear [min] * Torque [Nm], obserwowany próg około 11003.2
```

Te przedziały są opisem wzorców w zbiorze AI4I, a nie uniwersalnymi prawami fizycznymi dla wszystkich maszyn.

---

## 4. Przygotowanie danych i reprodukowalność

Podział:

```text
train = 8000
development/test = 2000
test_size = 0.2
random_state = 42
stratify = y
```

Rozkład klas:

```text
train: 7736 bez awarii / 264 awarie
test: 1934 bez awarii / 66 awarii
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

Nie skalować:

```text
Type_L
Type_M
```

`StandardScaler` jest dopasowywany tylko na `X_train_raw`, a następnie stosowany do train i test.

`03_data_preparation.ipynb` zapisuje:

```text
X_train_raw.csv
X_test_raw.csv
X_train_scaled.csv
X_test_scaled.csv
y_train.csv
y_test.csv
```

Każdy plik zachowuje oryginalny indeks jako:

```text
record_index
```

Poprawny sposób wczytywania:

```python
X_train = pd.read_csv(
    "../data/processed/X_train_scaled.csv",
    index_col="record_index"
)

X_test = pd.read_csv(
    "../data/processed/X_test_scaled.csv",
    index_col="record_index"
)

X_train_raw = pd.read_csv(
    "../data/processed/X_train_raw.csv",
    index_col="record_index"
)

X_test_raw = pd.read_csv(
    "../data/processed/X_test_raw.csv",
    index_col="record_index"
)

y_train = pd.read_csv(
    "../data/processed/y_train.csv",
    index_col="record_index"
).squeeze("columns")

y_test = pd.read_csv(
    "../data/processed/y_test.csv",
    index_col="record_index"
).squeeze("columns")
```

Zasady kontroli:

```python
assert X_train.index.equals(y_train.index)
assert X_test.index.equals(y_test.index)
assert X_train_raw.index.equals(y_train.index)
assert X_test_raw.index.equals(y_test.index)
assert "record_index" not in X_train.columns
assert "record_index" not in X_test.columns
```

---

## 5. Modele bazowe — aktualny spójny wynik

Modele:

```text
DummyClassifier(strategy="most_frequent")
LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
GradientBoostingClassifier(random_state=42)
```

Wspólny próg decyzyjny:

```text
DECISION_THRESHOLD = 0.5
predykcja klasy 1, gdy probability >= 0.5
```

Aktualny `results/model_results_baseline.csv`:

```text
model               accuracy  precision  recall   f1      roc_auc  pr_auc
Dummy               0.9670    0.0000     0.0000   0.0000  0.5000   0.0330
Log Reg             0.8475    0.1595     0.8485   0.2686  0.9202   0.4192
Random Forest       0.9895    0.8462     0.8333   0.8397  0.9910   0.8930
Gradient Boosting   0.9895    0.8814     0.7879   0.8320  0.9943   0.9145
```

Macierze pomyłek wynikające z aktualnych metryk:

```text
Dummy:
[[1934    0]
 [  66    0]]

Logistic Regression:
[[1639  295]
 [  10   56]]

Random Forest:
[[1924   10]
 [  11   55]]

Gradient Boosting:
[[1927    7]
 [  14   52]]
```

Interpretacja:

- Dummy potwierdza, że accuracy jest mylące przy rzadkiej klasie.
- Logistic Regression ma najwyższy recall, lecz generuje 295 fałszywych alarmów.
- Random Forest daje najlepszy kompromis między precision i recall.
- Gradient Boosting ma najlepszy ROC-AUC i PR-AUC oraz najwyższą precision, lecz niższy recall.

### 5.1. Ważna konwencja progu dla Random Forest

Historycznie `RandomForestClassifier.predict()` dawał 9 false positives. Aktualny kod tworzy klasy jawnie:

```python
y_pred = (y_proba >= 0.5).astype(int)
```

Rekord:

```text
record_index = 5653
random_forest_proba = 0.5
```

jest więc klasyfikowany jako awaria i staje się dziesiątym false positive.

To nie jest obecnie niespójność plików, lecz skutek jawnie wybranej konwencji:

```text
probability == threshold -> klasa 1
```

Przed threshold tuningiem należy zachować tę samą regułę we wszystkich funkcjach, tabelach i raportach.

---

## 6. False negatives

Liczba przeoczonych awarii:

```text
Logistic Regression: 10
Random Forest: 11
Gradient Boosting: 14
```

Przeoczone przez wszystkie trzy modele:

```text
8199, 8609, 4034, 9018, 7510
```

Rozkład trudnych przypadków:

```text
missed_by_n_models = 3: 5 rekordów
missed_by_n_models = 2: 6 rekordów
missed_by_n_models = 1: 8 rekordów
```

Grupy mechanizmów:

```text
PWF boundary: 442
HDF boundary: 3760, 3787, 3829, 4151, 4833, 4475
OSF bez jawnej interakcji: 1085, 2494, 9653, 9663, 8026
TWF wysokie rpm i wear: 2941, 9758
TWF wysokie wear: 7510, 9018, 4034, 8609, 8199
```

Hipoteza do kontrolowanego eksperymentu:

```text
Kryterium_OSF = Tool wear [min] * Torque [Nm]
```

Cecha nie jest bezpośrednim data leakage, ponieważ korzysta z parametrów dostępnych przed awarią. Jej wybór został jednak zainspirowany analizą etykiet OSF i musi być tak opisany w raporcie.

Aktualny eksport:

```text
results/false_negatives_baseline.csv
```

Plik zawiera prawdopodobieństwa zaokrąglone do 4 miejsc (pełna precyzja pozostaje w obliczeniach) oraz surowe parametry fizyczne trudnych rekordów.

---

## 7. False positives

Aktualny eksport:

```text
results/false_positives_baseline.csv
```

Liczności:

```text
Random Forest: 10
Gradient Boosting: 7
wspólne: 3764, 7677, 4862
```

Zakresy prawdopodobieństw:

```text
RF: 0.500–0.750
GB: 0.543–0.855
margin = predicted_probability - 0.5
```

Rekord graniczny:

```text
5653, Random Forest, probability = 0.5, margin = 0.0
```

Definicje podobieństwa do mechanizmów awarii:

```text
OSF_like: Tool wear * Torque >= 11003.2
HDF_like: temp diff w [7.6, 8.6] i rpm w [1212, 1379]
PWF_like: moc w jednym z dwóch przedziałów PWF
TWF_like: Tool wear w [198, 253]
```

Liczności flag:

```text
Random Forest:
HDF_like = 4
TWF_like = 6
OSF_like = 1
PWF_like = 0

Gradient Boosting:
HDF_like = 2
TWF_like = 5
OSF_like = 2
PWF_like = 0
```

Każdy zapisany false positive przypomina co najmniej jeden obszar awarii z EDA.

Interpretacja:

- mały margin oznacza przypadek blisko granicy decyzji,
- duży margin oznacza silny wzorzec przypominający awarię albo potencjalny problem etykiety,
- false positive nie zawsze oznacza bezsensowną predykcję; może być sensownym alarmem przy konserwatywnej polityce utrzymania ruchu.

Aktualny plik false positives zawiera prawdopodobieństwo, margin, flagi diagnostyczne oraz (od 2026-07-19) ponownie pełne surowe kolumny fizyczne; wartości raportowe są zaokrąglone do 4 miejsc.

---

## 8. Ocena zmian z 2026-07-14

### Mocne strony

- naprawiono zapis i odczyt `record_index`,
- rozdzielono dane surowe od skalowanych,
- ograniczono ryzyko przypadkowego użycia indeksu jako cechy,
- target jest prawidłowo jednowymiarowy,
- dodano asercje kontrolujące zgodność indeksów i wymiarów,
- funkcja `predict_with_threshold` centralizuje prawdopodobieństwa i decyzje,
- metryki, false negatives i false positives są generowane z tego samego przebiegu,
- analiza mechanizmów jest stosowana w jednolity sposób,
- notebooki mają logiczne nagłówki, cele i obserwacje,
- wyniki zostały zapisane poza outputami notebooka.

### Słabe strony i ukryte założenia

- `>= 0.5` powoduje inne zachowanie przy remisie niż historyczne `model.predict()` Random Forest,
- obecny zbiór testowy został użyty do porównania modeli, analizy błędów i projektowania kolejnych cech, więc pełni rolę development setu,
- reguły `*_like` są wyprowadzone z tego samego zbioru danych i nie są niezależnie zwalidowane,
- false-positive CSV stał się mniej samowystarczalny po usunięciu surowych parametrów,
- pełne prawdopodobieństwa zmiennoprzecinkowe zwiększają precyzję, ale pogarszają czytelność plików,
- nazewnictwo `Log Reg` i `Logistic Regression` jest niespójne,
- `PROJECT_CONTEXT.md` znajduje się w `.gitignore`, mimo że jest śledzony i ma być aktualizowany,
- bardzo dobre wyniki na AI4I nie gwarantują podobnej jakości na realnych danych przemysłowych.

Wniosek:

```text
Naprawa techniczna jest udana.
Obecny baseline można traktować jako spójny punkt odniesienia.
Największym ryzykiem nie jest już kod, lecz metodologia walidacji i przeuczenie decyzji projektowych do jednego development setu.
```

---

## 9. Metodologia dalszej pracy

Obecny zbiór 2000 rekordów wykorzystano do:

```text
porównania modeli
analizy false negatives
analizy false positives
projektowania cech
planowania threshold tuning
```

Dlatego nie jest już nietkniętym finalnym testem.

Preferowany schemat kolejnych eksperymentów:

```text
train
+ StratifiedKFold cross-validation
+ osobny final untouched test
```

Najbliższy kontrolowany eksperyment:

```text
A: obecne 8 cech
B: obecne 8 cech + Kryterium_OSF
```

Porównywać:

```text
precision
recall
F1
PR-AUC
ROC-AUC
średnią z walidacji krzyżowej
odchylenie standardowe
liczbę false negatives
liczbę false positives
```

Threshold tuning wykonać dopiero po wyborze modelu i zestawu cech. Próg dobierać na danych walidacyjnych, a nie na finalnym teście.

---

## 10. Zalecane następne zadania

1. Ujednolicić nazwy modeli w notebooku i CSV.
2. Dodać kontrolę, która jawnie raportuje rekordy z `probability == threshold`.
3. Zdecydować, czy konwencja ma być `>= threshold`, czy zgodna z `model.predict()`; potem stosować ją wszędzie.
4. Przywrócić surowe kolumny do `false_positives_baseline.csv` albo zapisać osobny pełny raport.
5. Zaokrąglać kopie raportowe prawdopodobieństw, pozostawiając pełną precyzję w obliczeniach.
6. Usunąć `PROJECT_CONTEXT.md` z `.gitignore`.
7. Dodać `requirements.txt`.
8. Rozbudować README o uruchamianie pipeline i najważniejsze wyniki.
9. Przenieść powtarzalne funkcje do skryptów w `python/` lub `src/`.
10. Wykonać eksperyment z `Kryterium_OSF` przy użyciu walidacji krzyżowej.
11. Dopiero potem wykonać threshold tuning.
12. W przyszłości wykonać sensitivity analysis dla 330 i 339 awarii.

---

## 11. Zasady dla kolejnych asystentów

```text
Najpierw przeczytaj PROJECT_CONTEXT.md.
Potem sprawdź 03_data_preparation.ipynb, 04_modeling.ipynb i pliki results/.
Nie używaj flag awarii, identyfikatorów ani record_index jako X.
Nie oceniaj modelu wyłącznie przez accuracy.
Pilnuj recall, precision, F1, PR-AUC i macierzy pomyłek.
Zachowuj rozróżnienie między danymi skalowanymi i surowymi.
Nie zmieniaj konwencji progu bez ponownego wygenerowania wszystkich wyników.
Oddzielaj fakty, interpretacje i hipotezy.
Nie dostrajaj modelu ani progu na finalnym teście.
Testuj pojedyncze zmiany, a dopiero potem je łącz.
Nie zakładaj, że reguły z AI4I są uniwersalnymi prawami przemysłowymi.
```
