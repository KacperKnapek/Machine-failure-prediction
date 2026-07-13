# PROJECT_CONTEXT.md

Ten plik jest trwałym kontekstem projektu predykcji awarii maszyn. Przy kolejnej pracy z repozytorium należy przeczytać go przed notebookami.

Ostatnia aktualizacja: 2026-07-13.

---

## 0. Najważniejsza zasada

Notebooki `.ipynb` mogą zostać zapisane bez outputów. Najważniejsze decyzje, wyniki i wnioski należy dlatego przechowywać tutaj oraz w plikach `results/` i `reports/`.

Nie używać jako cech modelu binarnego:

```text
TWF, HDF, PWF, OSF, RNF
```

Są to etykiety typów awarii i ich użycie prowadziłoby do data leakage.

---

## 1. Cel projektu

Projekt wykorzystuje **AI4I 2020 Predictive Maintenance Dataset** do klasyfikacji binarnej:

```text
Machine failure: 0 = brak awarii, 1 = awaria
```

Obecny projekt nie przewiduje bezpośrednio typu awarii. Flagi `TWF`, `HDF`, `PWF`, `OSF`, `RNF` służą do EDA i interpretacji błędów.

---

## 2. Struktura repozytorium

Najważniejsze pliki:

```text
README.md
.gitignore
PROJECT_CONTEXT.md

data/raw/produkcja.csv

data/processed/produkcja_clean.csv
data/processed/X_train_scaled.csv
data/processed/X_test_scaled.csv
data/processed/y_train.csv
data/processed/y_test.csv

notebooks/01_eda.ipynb
notebooks/02_data_cleaning.ipynb
notebooks/03_data_preparation.ipynb
notebooks/04_modeling.ipynb

python/cleaning.py
```

Stan dokumentacji:

```text
README.md nadal jest tylko krótkim szkicem i wymaga rozbudowy.
Nie utworzono jeszcze docelowych plików results/, reports/ ani requirements.txt.
```

---

## 3. Dane i najważniejsze wyniki EDA

Surowe dane:

```text
10000 rekordów
brak brakujących wartości
brak duplikatów
UDI i Product ID są unikalne
```

`Product ID` koduje `Type`, więc jest redundantny. `UDI` jest tylko identyfikatorem. Obie kolumny są usuwane przed modelowaniem.

Rozkład typu maszyny:

```text
L = 6000
M = 2997
H = 1003
```

Przed korektą etykiet:

```text
Machine failure = 1: 339 rekordów
Machine failure = 0: 9661 rekordów
```

Awaryjność według typu:

```text
H: 21 / 1003 = 2.09%
L: 235 / 6000 = 3.92%
M: 83 / 2997 = 2.77%
```

Typy awarii nie są rozłączne:

```text
24 rekordy mają co najmniej 2 typy awarii
1 rekord ma 3 typy awarii: UDI 5910, TWF + PWF + OSF
```

Problem RNF:

```text
RNF = 1 występuje w 19 rekordach
18 z nich ma Machine failure = 0 i nie ma innej flagi awarii
1 rekord łączy RNF z TWF i Machine failure = 1
```

Niespójność targetu:

```text
9 rekordów miało Machine failure = 1, ale wszystkie flagi typu awarii = 0
```

Przyjęta decyzja projektowa:

```text
ustawiono w nich Machine failure = 0
```

Jest to założenie, nie bezsporna prawda. W dalszej części projektu należy wykonać sensitivity analysis dla wariantu z 330 i 339 awariami.

### Charakterystyczne obszary awarii

```text
TWF:
Tool wear [min] = 198–253

HDF:
Temperature difference = 7.6–8.6
Rotational speed [rpm] = 1212–1379

PWF:
Power [W] = 1148.44–3477.24 albo 9004.43–10469.92

OSF:
ważna interakcja Tool wear [min] * Torque [Nm]
zaobserwowany próg około 11003.2 dla części danych
```

Wniosek:

```text
część mechanizmów awarii ma charakter nieliniowy i zależy od interakcji cech
```

---

## 4. Czyszczenie i feature engineering

`python/cleaning.py` oraz `02_data_cleaning.ipynb`:

1. Wczytują `data/raw/produkcja.csv`.
2. Tworzą:

```text
Power [W] = rpm * Torque * 2*pi/60
Temperature difference = Process temperature - Air temperature
```

3. Korygują 9 niespójnych etykiet.
4. Usuwają:

```text
UDI
Product ID
Air temperature [K]
TWF
HDF
PWF
OSF
RNF
```

5. Kodują `Type` przez one-hot encoding:

```text
Type_L
Type_M
```

Przetworzony zbiór:

```text
shape = (10000, 9)
8 cech + Machine failure
Machine failure = 1: 330
Machine failure = 0: 9670
```

---

## 5. Przygotowanie danych

`03_data_preparation.ipynb`:

```text
train = 8000 rekordów
current test = 2000 rekordów
random_state = 42
stratify = y
```

Rozkład klas:

```text
train: 7736 braków awarii, 264 awarie
test: 1934 braki awarii, 66 awarii
```

Skalowane kolumny:

```text
Process temperature [K]
Temperature difference
Rotational speed [rpm]
Torque [Nm]
Power [W]
Tool wear [min]
```

`StandardScaler` został dopasowany tylko na `X_train`, więc na tym etapie nie wystąpił klasyczny leakage ze zbioru testowego.

---

## 6. Modele bazowe

Modele w `04_modeling.ipynb`:

```text
DummyClassifier
LogisticRegression(class_weight='balanced')
RandomForestClassifier(n_estimators=200, class_weight='balanced')
GradientBoostingClassifier
```

Wyniki przy progu 0.5:

```text
model                 accuracy  precision  recall   f1      roc_auc  pr_auc
Dummy                 0.9670    0.0000     0.0000   0.0000  0.5000   0.0330
Logistic Regression   0.8475    0.1595     0.8485   0.2686  0.9202   0.4192
Random Forest         0.9900    0.8594     0.8333   0.8462  0.9910   0.8930
Gradient Boosting     0.9895    0.8814     0.7879   0.8320  0.9943   0.9145
```

Macierze pomyłek:

```text
Dummy:
[[1934, 0],
 [66,   0]]

Logistic Regression:
[[1639, 295],
 [10,    56]]

Random Forest:
[[1925, 9],
 [11,  55]]

Gradient Boosting:
[[1927, 7],
 [14,  52]]
```

Interpretacja:

```text
Random Forest ma najlepszy aktualny kompromis precision/recall/F1.
Gradient Boosting ma najlepszy ROC-AUC i PR-AUC, lecz niższy recall.
Logistic Regression ma wysoki recall, ale generuje 295 false positives.
Accuracy sama w sobie jest myląca z powodu silnego niezbalansowania klas.
```

---

## 7. False negatives

Liczba przeoczonych awarii:

```text
Logistic Regression: 10
Random Forest: 11
Gradient Boosting: 14
```

Nakładanie błędów:

```text
47 awarii wykryły wszystkie modele
8 awarii przeoczył 1 model
6 awarii przeoczyły 2 modele
5 awarii przeoczyły wszystkie 3 modele
```

Przeoczone przez wszystkie modele:

```text
8199, 8609, 4034, 9018, 7510
```

Wszystkie trudne rekordy:

```text
3 modele: 8199, 8609, 4034, 9018, 7510
2 modele: 9758, 4475, 8026, 9663, 2941, 4833
1 model: 4151, 9653, 2494, 1085, 3829, 3760, 442, 3787
```

Robocze grupy mechanizmów:

```text
PWF_boundary_case:
442

HDF_boundary_case:
3760, 3787, 3829, 4151, 4833, 4475

OSF_missing_engineered_criterion:
1085, 2494, 9653, 9663, 8026

TWF_high_rotational_speed_and_tool_wear:
2941, 9758

TWF_high_tool_wear:
7510, 9018, 4034, 8609, 8199
```

Najważniejsza hipoteza:

```text
Kryterium_OSF = Tool wear [min] * Torque [Nm]
```

Może poprawić część błędów OSF, ponieważ obecne modele widzą dwie składowe osobno, ale nie mają jawnie podanej interakcji.

---

## 8. False positives

Analiza została wykonana dla Random Forest i Gradient Boosting przy:

```text
threshold = 0.5
```

Liczba błędów:

```text
Random Forest: 9
Gradient Boosting: 7
```

Indeksy Random Forest:

```text
3764, 4898, 8198, 7081, 7677, 4115, 4110, 7255, 4862
```

Indeksy Gradient Boosting:

```text
6925, 3764, 7677, 1507, 7935, 4862, 9671
```

Wspólne false positives:

```text
3764, 7677, 4862
```

Zakresy prawdopodobieństw:

```text
Random Forest: 0.53–0.75
Gradient Boosting: 0.543–0.855
```

Zdefiniowany margines:

```text
margin = predicted_probability - 0.5
```

Random Forest:

```text
index  proba  margin
3764   0.670  0.170
4898   0.610  0.110
8198   0.560  0.060
7081   0.640  0.140
7677   0.645  0.145
4115   0.580  0.080
4110   0.750  0.250
7255   0.530  0.030
4862   0.665  0.165
```

Gradient Boosting:

```text
index  proba  margin
6925   0.742  0.242
3764   0.543  0.043
7677   0.855  0.355
1507   0.850  0.350
7935   0.589  0.089
4862   0.548  0.048
9671   0.657  0.157
```

Wstępne grupy:

```text
HDF-like / blisko granicy:
3764, 4115, 4110, 4862

OSF-like / wysokie Tool wear * Torque:
4898, 7081, 7677, 6925, 1507, 9671

TWF-like / wysokie Tool wear:
8198, 7255, 7935
```

Interpretacja:

```text
Część false positives ma mały margin i leży tuż nad progiem 0.5.
Inne, zwłaszcza 7677 i 1507 dla Gradient Boosting, mają wysokie prawdopodobieństwo i przypominają modelowi silny wzorzec awarii.
False positives często leżą blisko znanych regionów HDF, OSF lub TWF, ale po stronie etykiety Machine failure = 0.
```

Do dokończenia w notebooku:

```text
1. policzyć Kryterium_OSF dla wszystkich false positives
2. dodać kolumnę mechanism_group
3. stworzyć jedną zwartą tabelę obu modeli
4. dopisać końcowy wniosek Markdown
```

---

## 9. Ważna korekta metodologiczna

Obecny `X_test / y_test` został wykorzystany do:

```text
porównania modeli
analizy false negatives rekord po rekordzie
analizy false positives rekord po rekordzie
formułowania hipotez nowych cech
planowania threshold tuningu
```

W praktyce przestał być nietkniętym finalnym test setem i stał się zbiorem deweloperskim.

Zasada dalszej pracy:

```text
Nie wybierać nowych cech, hiperparametrów ani progu na finalnym teście.
Przed końcową oceną przygotować nowy holdout albo schemat train/validation/test.
```

Preferowany schemat:

```text
train + StratifiedKFold cross-validation
final untouched test
```

Alternatywnie:

```text
train / validation / final test
```

Finalny test uruchomić tylko raz po wyborze modelu, cech i progu.

---

## 10. Plan na dzisiaj

### Etap A — zamknąć analizę błędów w `04_modeling.ipynb`

1. Dodać do tabel false positives:

```python
Kryterium_OSF = Tool wear [min] * Torque [Nm]
```

2. Utworzyć jedną tabelę z kolumnami:

```text
index
model
predicted_probability
margin
Temperature difference
Rotational speed [rpm]
Tool wear [min]
Torque [Nm]
Kryterium_OSF
mechanism_group
```

3. Przypisać grupy:

```text
HDF_boundary_like
OSF_boundary_like
TWF_like
unclear
```

4. Dopisać w notebooku końcowe wnioski oddzielające:

```text
Fakty
Interpretacje
Hipotezy
```

### Etap B — zachować wyniki poza notebookiem

Utworzyć:

```text
results/model_results_baseline.csv
results/false_negatives_baseline.csv
results/false_positives_baseline.csv
```

### Etap C — przygotować następny eksperyment

Proponowany notebook:

```text
notebooks/05_feature_engineering_and_validation.ipynb
```

Najpierw ustalić poprawny schemat walidacji, a następnie porównać:

```text
A: obecne 8 cech
B: obecne 8 cech + Kryterium_OSF
```

Porównywać:

```text
mean precision
mean recall
mean F1
mean PR-AUC
odchylenie standardowe metryk
udział false negatives
```

Nie dodawać od razu wielu cech. Najpierw sprawdzić pojedynczą hipotezę.

### Etap D — threshold tuning dopiero po feature engineering

Dla wybranego modelu i zestawu cech przygotować tabelę:

```text
threshold  precision  recall  f1  false_positives  false_negatives
```

Przykładowe kryteria wyboru progu:

```text
maksymalizacja F1
najwyższa precision przy recall >= 0.90
minimalizacja koszt_FN * FN + koszt_FP * FP
```

Nie zakładać, że `0.5` jest najlepszym progiem.

---

## 11. Dalsza kolejność projektu

Po kontrolowanym eksperymencie z cechą i progiem:

1. Permutation importance.
2. Ewentualnie SHAP.
3. Sensitivity analysis dla 9 zmienionych etykiet.
4. Rozbudowa README.
5. Dodanie `requirements.txt`.
6. Przeniesienie powtarzalnego kodu do `python/modeling.py` lub pipeline scikit-learn.
7. Finalna ocena na nietkniętym teście.

---

## 12. Zasady dla kolejnych asystentów

Przed pracą sprawdzić:

```text
PROJECT_CONTEXT.md
python/cleaning.py
notebooks/02_data_cleaning.ipynb
notebooks/03_data_preparation.ipynb
notebooks/04_modeling.ipynb
```

Najważniejsze reguły:

```text
Nie używać flag typów awarii jako X.
Nie używać UDI ani Product ID jako X.
Nie oceniać modelu wyłącznie przez accuracy.
Pilnować recall, precision, F1 i PR-AUC klasy awarii.
Oddzielać fakty, interpretacje i hipotezy.
Nie stroić modelu na finalnym teście.
Obecny test traktować jako zbiór deweloperski.
Najpierw testować pojedyncze zmiany, potem łączyć je w finalny model.
```
