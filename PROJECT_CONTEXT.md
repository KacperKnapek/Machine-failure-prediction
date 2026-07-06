# PROJECT_CONTEXT.md

Ten plik jest stałym kontekstem projektu dla Kacpra oraz narzędzi AI pracujących z repozytorium. Ma ograniczyć konieczność ciągłego tłumaczenia, co znajduje się w projekcie, jaki jest cel analizy, jakie decyzje zostały już podjęte i jakie wyniki należy zachować nawet wtedy, gdy notebooki zostaną zapisane bez `outputs`.

Ostatnia aktualizacja kontekstu: 2026-07-06.

---

## 0. Najważniejsza informacja dla AI

Przy każdej kolejnej pracy z tym repozytorium najpierw przeczytaj ten plik, a dopiero potem zaglądaj do notebooków i skryptów.

Ważne rozróżnienie:

```text
GitHub connector może pokazywać notebooki .ipynb bez outputs, nawet jeżeli użytkownik lokalnie ma wyniki komórek.
Najważniejsze wyniki z EDA, czyszczenia, przygotowania danych i pierwszego modelowania są zapisane w tym pliku.
```

Dlatego `PROJECT_CONTEXT.md` ma być źródłem trwałego kontekstu, jeśli notebooki zostaną wyczyszczone przez `Clear All Outputs`.

---

## 1. Cel projektu

Projekt dotyczy predykcji awarii maszyn na podstawie danych produkcyjnych / utrzymania ruchu.

Główny cel modelowania:

```text
Przewidywanie zmiennej binarnej: Machine failure
```

Czyli model ma odpowiedzieć na pytanie:

```text
Czy maszyna ulegnie awarii? 0 = nie, 1 = tak
```

Aktualny kierunek projektu to klasyfikacja binarna, a nie pełna klasyfikacja wieloetykietowa typów awarii.

Typy awarii `TWF`, `HDF`, `PWF`, `OSF`, `RNF` są ważne interpretacyjnie, ale nie powinny być używane jako zwykłe cechy wejściowe do modelu przewidującego `Machine failure`, bo prowadziłoby to do przecieku danych.

---

## 2. Dane wejściowe

Surowy plik danych:

```text
data/raw/produkcja.csv
```

Oryginalne kolumny:

```text
UDI
Product ID
Type
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
Machine failure
TWF
HDF
PWF
OSF
RNF
```

Typy awarii:

```text
TWF = Tool Wear Failure
HDF = Heat Dissipation Failure
PWF = Power Failure
OSF = Overstrain Failure
RNF = Random Failure
```

Znaczenie głównych kolumn:

- `UDI` — unikalny identyfikator rekordu.
- `Product ID` — identyfikator produktu / maszyny; pierwsza litera koduje `Type`.
- `Type` — typ maszyny: H, L, M.
- `Air temperature [K]` — temperatura powietrza w kelwinach.
- `Process temperature [K]` — temperatura procesu w kelwinach.
- `Rotational speed [rpm]` — prędkość obrotowa.
- `Torque [Nm]` — moment obrotowy.
- `Tool wear [min]` — zużycie narzędzia w minutach.
- `Machine failure` — główna zmienna docelowa.
- `TWF`, `HDF`, `PWF`, `OSF`, `RNF` — flagi typów awarii.

---

## 3. Obecna struktura projektu rozpoznana w repozytorium

Ważne pliki:

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

Uwaga techniczna: dostępne narzędzie GitHub może nie mieć wygodnej funkcji pełnego listowania katalogów. Ten spis należy aktualizować ręcznie, gdy pojawią się nowe istotne pliki.

---

## 4. Najważniejsze wyniki EDA

### 4.1. Rozmiar i kompletność danych po dodaniu cech inżynierskich

Po dodaniu `Temperature difference` i `Power [W]` zbiór w EDA miał:

```text
shape = (10000, 16)
```

Braki danych:

```text
df.isnull().any().any() = False
```

Duplikaty:

```text
df.duplicated().sum() = 0
```

Identyfikatory:

```text
UDI is_unique = True
Product ID is_unique = True
```

Interpretacja:

```text
Dane są kompletne, bez duplikatów i mają unikalne identyfikatory.
UDI oraz Product ID nie powinny być używane jako cechy predykcyjne.
```

### 4.2. Product ID koduje Type

Wynik `pd.crosstab(df['Type'], df['Product ID'].str[0])`:

```text
Product ID     H     L     M
Type
H           1003     0     0
L              0  6000     0
M              0     0  2997
```

Interpretacja:

```text
Pierwsza litera Product ID jednoznacznie odpowiada Type.
Product ID jest redundantny względem Type i nie powinien być używany jako cecha modelu.
```

### 4.3. Rozkład typów maszyn

```text
L    6000
M    2997
H    1003
```

Interpretacja:

```text
Najwięcej jest maszyn typu L, potem M, najmniej H.
Przy porównywaniu liczby awarii między typami trzeba patrzeć na procenty, a nie tylko liczby bezwzględne.
```

### 4.4. Rozkład targetu w surowym EDA

Przed korektą niespójnych rekordów:

```text
Machine failure = 1: 339 rekordów
Machine failure = 0: 96.61%
Machine failure = 1: 3.39%
```

Interpretacja:

```text
Zbiór jest silnie niezbalansowany.
Accuracy może być złudnie wysoka, jeśli model będzie prawie zawsze przewidywał brak awarii.
```

### 4.5. Awaryjność według typu maszyny

Liczba awarii według typu:

```text
L: 235
M: 83
H: 21
```

Tabela liczności `Type` × `Machine failure`:

```text
Machine failure     0    1
Type
H                 982   21
L                5765  235
M                2914   83
```

Awaryjność procentowa w obrębie typu:

```text
H: 2.09%
L: 3.92%
M: 2.77%
```

Interpretacja:

```text
Największą względną awaryjność ma typ L.
Nie wystarczy patrzeć na bezwzględne liczby awarii, bo typ L ma też najwięcej rekordów.
```

### 4.6. Suma typów awarii według typu maszyny

```text
Type  Machine failure  TWF  HDF  PWF  OSF  RNF
H                  21    7    8    5    2    4
L                 235   25   76   59   87   13
M                  83   14   31   31    9    2
```

Interpretacja robocza:

```text
Dla typu L szczególnie często pojawia się OSF oraz HDF.
Dla typu M istotne są HDF i PWF.
Dla typu H liczby są małe, więc wnioski trzeba traktować ostrożnie.
```

### 4.7. Rekordy z kilkoma typami awarii naraz

Rekordy, gdzie suma flag `TWF`, `HDF`, `PWF`, `OSF`, `RNF` jest większa lub równa 2:

```text
24 rekordy mają co najmniej dwa typy awarii jednocześnie.
```

Dla co najmniej trzech typów awarii:

```text
1 rekord ma co najmniej trzy typy awarii jednocześnie.
UDI = 5910, Product ID = H35323, Type = H
TWF = 1, PWF = 1, OSF = 1
Machine failure = 1
```

Interpretacja:

```text
Typy awarii nie są rozłączne. Jeden rekord może należeć do kilku kategorii awarii.
To wzmacnia decyzję, że projekt binarny Machine failure jest prostszy niż predykcja pełnego typu awarii.
```

### 4.8. Problem RNF

Dla `RNF == 1`:

```text
RNF = 1 występuje w 19 rekordach.
```

Rekordy, gdzie istnieje dowolna flaga awarii, ale `Machine failure == 0`:

```text
18 rekordów ma jakąś flagę awarii przy Machine failure = 0.
Wszystkie te 18 rekordów to przypadki RNF = 1 bez innych typów awarii.
```

Jeden przypadek RNF współwystępuje z inną awarią:

```text
UDI = 3612
Type = L
Machine failure = 1
TWF = 1
RNF = 1
```

Interpretacja:

```text
RNF jest problematyczne interpretacyjnie.
Większość RNF wygląda jak losowe zdarzenie bez potwierdzenia w Machine failure.
RNF nie powinno być traktowane tak samo jak deterministyczne typy awarii.
```

### 4.9. Machine failure bez żadnego typu awarii

Przed czyszczeniem znaleziono:

```text
9 rekordów, gdzie Machine failure = 1, ale TWF = HDF = PWF = OSF = RNF = 0.
```

Po korekcie w `02_data_cleaning.ipynb` wynik sprawdzenia wynosi:

```text
0 takich rekordów
```

Decyzja:

```text
Te rekordy potraktowano jako niespójność etykiety / szum i ustawiono Machine failure = 0.
To jest założenie projektowe, a nie prawda absolutna.
```

Kontrargument:

```text
Mogły to być rzeczywiste awarie, ale bez poprawnie oznaczonego typu awarii.
W raporcie trzeba jawnie napisać, że jest to arbitralna, ale uzasadniona decyzja czyszczenia danych.
```

### 4.10. Awaria przy Tool wear = 0

W EDA znaleziono 3 rekordy:

```text
Tool wear [min] = 0
Machine failure = 1
```

Wszystkie trzy przypadki mają:

```text
PWF = 1
```

Ich moce:

```text
9239.21 W
9177.81 W
9432.38 W
```

Interpretacja:

```text
Awaria przy zerowym zużyciu narzędzia nie musi być błędem, jeśli jest to Power Failure.
Wysoka moc może tłumaczyć awarię niezależnie od Tool wear.
```

### 4.11. Korelacje temperatur

```text
Air temperature [K] vs Process temperature [K] = 0.876107
Air temperature [K] vs Temperature difference = -0.699583
Process temperature [K] vs Temperature difference = -0.268413
```

Interpretacja:

```text
Air temperature i Process temperature są silnie dodatnio skorelowane.
Temperature difference wnosi inną informację, bo jest różnicą między temperaturą procesu i powietrza.
Usunięcie Air temperature [K] po utworzeniu Temperature difference jest metodologicznie uzasadnione, ale jest to decyzja modelowa, nie konieczność matematyczna.
```

### 4.12. Charakterystyczne przedziały dla typów awarii

#### TWF

Dla `TWF == 1`:

```text
Tool wear [min]: min 198, max 253
```

Interpretacja:

```text
TWF jest silnie związany z dużym zużyciem narzędzia.
```

#### HDF

Dla `HDF == 1`:

```text
Temperature difference: min 7.6, max 8.6
Rotational speed [rpm]: min 1212, max 1379
```

Interpretacja:

```text
HDF pojawia się przy niskiej różnicy temperatur i relatywnie niskich obrotach.
To sugeruje mechanizm związany z odprowadzaniem ciepła.
```

#### PWF

Dla `PWF == 1` analiza K-Means na `Power [W]` wskazała dwa przedziały:

```text
Przedział 1: 1148.44–3477.24 W
Przedział 2: 9004.43–10469.92 W
```

Interpretacja:

```text
PWF może oznaczać awarie zarówno przy zbyt niskiej, jak i zbyt wysokiej mocy.
To ważny nieliniowy wzorzec: samo 'im więcej mocy, tym gorzej' jest zbyt prostym założeniem.
```

#### OSF

Dla `OSF == 1` wyznaczono próg na iloczynie:

```text
Tool wear [min] * Torque [Nm] ≈ 11003.2
```

Interpretacja:

```text
OSF jest dobrze opisywany przez interakcję zużycia narzędzia i momentu obrotowego.
To wskazuje, że sama pojedyncza cecha może nie wystarczyć; ważne są zależności między cechami.
```

---

## 5. Czyszczenie danych w `python/cleaning.py` i `02_data_cleaning.ipynb`

Skrypt `python/cleaning.py` wykonuje następujący proces:

1. Wczytuje dane z:

```text
./data/raw/produkcja.csv
```

2. Oblicza `Power [W]`:

```text
Power [W] = Rotational speed [rpm] * Torque [Nm] * (2 * pi / 60)
```

3. Oblicza `Temperature difference`:

```text
Temperature difference = Process temperature [K] - Air temperature [K]
```

4. Koryguje niespójne rekordy `Machine failure == 1` bez żadnej flagi typu awarii.

5. Usuwa kolumny:

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

6. Koduje `Type` przez one-hot encoding z `drop_first=True` i `dtype=int`.

Powstają kolumny:

```text
Type_L
Type_M
```

Typ pominięty przez `drop_first=True` jest kategorią bazową.

7. Zapisuje wynik do:

```text
./data/processed/produkcja_clean.csv
```

---

## 6. Dane po przetworzeniu

Plik przetworzony:

```text
data/processed/produkcja_clean.csv
```

W `03_data_preparation.ipynb` potwierdzono:

```text
shape = (10000, 9)
```

Kolumny:

```text
Type_L
Type_M
Process temperature [K]
Temperature difference
Rotational speed [rpm]
Torque [Nm]
Power [W]
Tool wear [min]
Machine failure
```

Interpretacja:

```text
Machine failure pozostaje targetem.
Pozostałe 8 kolumn to cechy wejściowe X.
```

Po czyszczeniu liczba awarii wynosi:

```text
Machine failure = 1: 330 rekordów
Machine failure = 0: 9670 rekordów
```

Proporcje:

```text
Machine failure = 0: 96.7%
Machine failure = 1: 3.3%
```

---

## 7. Przygotowanie danych pod ML z `03_data_preparation.ipynb`

Podział na cechy i target:

```text
X shape = (10000, 8)
y shape = (10000,)
```

Podział train/test:

```text
test_size = 0.2
random_state = 42
stratify = y
```

Wynik podziału:

```text
Train: 8000 rekordów
Test: 2000 rekordów
Awarie w train: 264 (3.3%)
Awarie w test: 66 (3.3%)
```

Skalowane kolumny numeryczne:

```text
Process temperature [K]
Temperature difference
Rotational speed [rpm]
Torque [Nm]
Power [W]
Tool wear [min]
```

Nie skalować one-hot encoded:

```text
Type_L
Type_M
```

Schemat skalowania:

```text
StandardScaler fitowany tylko na X_train.
Ten sam scaler zastosowany do X_train i X_test.
Nie ma przecieku informacji z testu do treningu.
```

Parametry `StandardScaler` z train:

```text
mean_:
Process temperature [K] = 310.005375
Temperature difference = 10.0002375
Rotational speed [rpm] = 1538.92525
Torque [Nm] = 39.98705
Power [W] = 6280.70232
Tool wear [min] = 107.59825

scale_:
Process temperature [K] = 1.48058303
Temperature difference = 0.999674294
Rotational speed [rpm] = 178.987704
Torque [Nm] = 9.96722403
Power [W] = 1068.53168
Tool wear [min] = 63.5351269
```

`03_data_preparation.ipynb` zapisuje pliki:

```text
data/processed/X_train_scaled.csv
data/processed/X_test_scaled.csv
data/processed/y_train.csv
data/processed/y_test.csv
```

`04_modeling.ipynb` korzysta właśnie z tych plików.

---

## 8. Aktualny stan modelowania z `04_modeling.ipynb`

### 8.1. Dane wejściowe do modelowania

`04_modeling.ipynb` wczytuje:

```text
X_train = data/processed/X_train_scaled.csv
X_test = data/processed/X_test_scaled.csv
y_train = data/processed/y_train.csv
y_test = data/processed/y_test.csv
```

Potwierdzone rozmiary:

```text
X_train shape = (8000, 8)
X_test shape = (2000, 8)
y_train shape = (8000,)
y_test shape = (2000,)
```

Rozkład klas:

```text
y_train:
0 = 7736
1 = 264

y_test:
0 = 1934
1 = 66
```

### 8.2. Funkcje pomocnicze

Notebook zawiera funkcje:

```text
evaluate_model(name, model, X_test, y_test)
get_metrics(name, model, X_test, y_test)
get_false_negatives(model)
```

Metryki liczone w notebooku:

```text
accuracy
precision
recall
f1
roc_auc
pr_auc / average_precision_score
confusion_matrix
classification_report
```

### 8.3. Modele porównane w pierwszym modelowaniu

Porównano:

```text
DummyClassifier(strategy='most_frequent')
LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
GradientBoostingClassifier(random_state=42)
```

Tabela wyników:

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
[[1934    0]
 [  66    0]]

Logistic Regression:
[[1639  295]
 [  10   56]]

Random Forest:
[[1925    9]
 [  11   55]]

Gradient Boosting:
[[1927    7]
 [  14   52]]
```

Interpretacja wyników:

```text
Dummy pokazuje, dlaczego accuracy jest mylące: 96.7% accuracy przy zerowym wykrywaniu awarii.
Logistic Regression ma bardzo wysoki recall dla awarii, ale generuje bardzo dużo false positives.
Random Forest ma najlepszy F1 dla klasy awarii i bardzo dobry kompromis precision/recall.
Gradient Boosting ma najlepszy ROC-AUC i PR-AUC oraz najwyższą precision, ale niższy recall niż Random Forest i Logistic Regression.
```

Roboczy wniosek:

```text
Najlepszym kandydatem bazowym jest Random Forest, jeśli priorytetem jest równowaga między precision, recall i F1.
Gradient Boosting jest bardzo mocny rankingowo po prawdopodobieństwach, więc warto później zbadać threshold tuning.
Logistic Regression jest użyteczna jako model interpretowalny i jako kontrast, ale w aktualnej postaci daje zbyt wiele fałszywych alarmów.
```

Kontrargument metodologiczny:

```text
Wyniki są bardzo dobre, ale zbiór AI4I ma częściowo regułowy charakter awarii.
Nie wolno od razu zakładać, że model będzie równie dobry na realnych danych przemysłowych.
Potrzebna jest walidacja, analiza stabilności i test na danych spoza tego samego generatora / rozkładu, jeśli takie dane będą dostępne.
```

---

## 9. Analiza przeoczonych awarii — False Negatives

### 9.1. Liczba false negatives według modelu

```text
Logistic Regression FN: 10
Random Forest FN: 11
Gradient Boosting FN: 14
```

Z 66 awarii w teście:

```text
missed_by_n_models = 0: 47 rekordów
missed_by_n_models = 1: 8 rekordów
missed_by_n_models = 2: 6 rekordów
missed_by_n_models = 3: 5 rekordów
```

Nakładanie się false negatives:

```text
Logistic Regression ∩ Random Forest: 7
Logistic Regression ∩ Gradient Boosting: 5
Random Forest ∩ Gradient Boosting: 9
All three models: 5
```

Awarie przeoczone przez wszystkie trzy modele:

```text
{8609, 4034, 8199, 7510, 9018}
```

### 9.2. Wszystkie trudne przypadki z testu, które przeoczył co najmniej jeden model

Rekordy uporządkowane według liczby modeli, które się pomyliły:

```text
missed_by_n_models = 3:
8199, 8609, 4034, 9018, 7510

missed_by_n_models = 2:
9758, 4475, 8026, 9663, 2941, 4833

missed_by_n_models = 1:
4151, 9653, 2494, 1085, 3829, 3760, 442, 3787
```

Uwaga: indeksy powyżej są indeksami rekordów w danych, a nie wartościami `UDI`. Dla przykładu indeks `8199` ma `UDI = 8200`.

### 9.3. Predykcje modeli dla trudnych przypadków

```text
index  log_reg_pred  random_forest_pred  gradient_boosting_pred  missed_by_n_models
8199   0             0                   0                       3
8609   0             0                   0                       3
4034   0             0                   0                       3
9018   0             0                   0                       3
7510   0             0                   0                       3
9758   1             0                   0                       2
4475   0             0                   1                       2
8026   1             0                   0                       2
9663   1             0                   0                       2
2941   1             0                   0                       2
4833   0             0                   1                       2
4151   0             1                   1                       1
9653   1             1                   0                       1
2494   1             1                   0                       1
1085   1             1                   0                       1
3829   1             1                   0                       1
3760   1             1                   0                       1
442    0             1                   1                       1
3787   0             1                   1                       1
```

### 9.4. Surowe parametry trudnych przypadków

```text
index  Type  failure_type  Rotational speed  Torque  Tool wear  Power [W]  Temp diff  Kryterium_OSF
3787   L     HDF           1377              47.3    22         6820.62    8.5        1040.6
442    L     PWF           1399              61.5    61         9009.93    11.1       3751.5
3760   L     HDF           1377              46.8    166        6748.52    8.6        7768.8
3829   H     HDF           1366              48.4    130        6923.48    8.6        6292.0
1085   L     OSF           1385              56.4    202        8180.08    10.8       11392.8
2494   L     OSF           1329              53.6    207        7459.65    9.7        11095.2
9653   L     OSF           1373              55.7    201        8008.56    10.9       11195.7
4151   M     HDF           1373              48.0    73         6901.45    8.5        3504.0
4833   L     HDF           1377              41.6    34         5998.68    8.5        1414.4
2941   M     TWF           1996              19.8    203        4138.61    8.9        4019.4
9663   L     OSF           1435              48.8    229        7333.32    11.0       11175.2
8026   L     OSF           1399              50.2    222        7354.45    11.2       11144.4
4475   L     HDF           1351              41.8    10         5913.71    7.8        418.0
9758   L     TWF           2271              16.2    218        3852.66    11.2       3531.6
7510   L     TWF           1524              38.9    214        6208.16    11.3       8324.6
9018   L     TWF           1615              35.4    217        5986.93    10.8       7681.8
4034   L     TWF           1615              29.0    235        4904.55    8.8        6815.0
8609   L     TWF           1475              40.5    222        6255.70    10.9       8991.0
8199   L     TWF           1737              27.0    225        4911.25    11.5       6075.0
```

### 9.5. Interpretacyjne grupy mechanizmów trudnych przypadków

Robocze grupowanie zapisane w notebooku:

```python
mechanism_groups = {
    "PWF_boundary_case": [442],
    "HDF_boundary_case": [3760, 3787, 3829, 4151, 4833, 4475],
    "OSF_missing_engineered_criterion": [1085, 2494, 9653, 9663, 8026],
    "TWF_high_rotational_speed_and_tool_wear": [2941, 9758],
    "TWF_high_tool_wear": [7510, 9018, 4034, 8609, 8199]
}
```

Opis użytkownika / interpretacja robocza:

```text
442 — bardzo blisko dolnego krańca przedziału dla PWF.

3760, 3787, 3829, 4151, 4833, 4475 — bardzo blisko krańców przedziału dla obu parametrów HDF: Rotational speed i Temperature difference.

1085, 2494, 9653, 9663, 8026 — algorytmy nie widziały jawnie Kryterium_OSF, czyli Tool wear [min] * Torque [Nm].

2941, 9758 — wysokie Rotational speed oraz Tool wear przy TWF; w EDA nie znaleziono prostej zależności między wysokim Rotational speed a TWF.

7510, 9018, 4034, 8609, 8199 — wysokie Tool wear przy TWF; to są przypadki przeoczone przez wszystkie trzy modele.
```

Interpretacja metodologiczna:

```text
Część false negatives wygląda jak przypadki brzegowe reguł awarii.
Modele drzewiaste i boosting dobrze działają ogólnie, ale nie zawsze łapią rekordy leżące blisko granic mechanizmów awarii.
Logistic Regression przy class_weight='balanced' łapie dużo awarii, ale płaci za to ogromną liczbą false positives.
```

Hipoteza do sprawdzenia:

```text
Dodanie jawnych cech interakcyjnych może poprawić wykrywanie części trudnych przypadków, zwłaszcza OSF.
Najważniejszy kandydat: Kryterium_OSF = Tool wear [min] * Torque [Nm].
```

Kontrargument:

```text
Kryterium_OSF zostało zauważone podczas analizy typów awarii. Jeżeli cecha została zaprojektowana na podstawie wiedzy o etykietach typów awarii, trzeba jasno opisać to w raporcie jako feature engineering inspirowany EDA.
Sama cecha nie jest bezpośrednim data leakage, bo używa tylko parametrów dostępnych przed awarią, ale decyzja o jej dodaniu pochodzi z analizy etykiet.
```

---

## 10. Decyzje metodologiczne, które trzeba zachować

### 10.1. Data leakage

Nie używać jako cech wejściowych do modelu binarnego:

```text
TWF
HDF
PWF
OSF
RNF
```

Powód:

```text
Te kolumny opisują rodzaj awarii, więc zawierają informację bardzo bliską targetowi Machine failure.
Użycie ich jako X dałoby sztucznie dobry wynik i byłoby klasycznym data leakage.
```

### 10.2. Identyfikatory

Nie używać jako cech wejściowych:

```text
UDI
Product ID
```

Powód:

```text
UDI jest tylko numerem rekordu.
Product ID jest unikalny i koduje Type, więc grozi zapamiętywaniem oraz redundancją.
```

### 10.3. Klasy niezbalansowane

Awarii jest tylko około 3.3% po czyszczeniu.

Nie oceniać modelu wyłącznie przez:

```text
accuracy
```

Ważniejsze metryki:

```text
recall dla klasy 1
precision dla klasy 1
F1-score dla klasy 1
confusion matrix
ROC-AUC
PR-AUC / Average Precision
```

Szczególnie ważny jest `recall` dla klasy awarii, bo w predykcyjnym utrzymaniu ruchu przeoczenie awarii bywa kosztowniejsze niż fałszywy alarm.

### 10.4. Korekta 9 rekordów bez typu awarii

Fakty:

```text
Przed korektą: 339 awarii.
Po korekcie: 330 awarii.
```

Interpretacja przyjęta w projekcie:

```text
Brak jakiejkolwiek flagi typu awarii oznacza brak wiarygodnego potwierdzenia awarii.
```

Kontrargument:

```text
Mogły to być rzeczywiste awarie, ale z błędnie nieuzupełnionym typem.
```

W dalszym raporcie warto jawnie napisać, że jest to arbitralna, ale uzasadniona decyzja czyszczenia danych.

---

## 11. Notebooki

### `notebooks/01_eda.ipynb`

Notebook do eksploracyjnej analizy danych.

Zawiera między innymi:

- import `pandas`, `matplotlib`, `seaborn`, `numpy`,
- wczytanie `../data/raw/produkcja.csv`,
- funkcję pomocniczą `plot_box_hist`,
- analizę struktury danych,
- analizę braków, duplikatów i typów danych,
- analizę typów maszyn,
- analizę RNF i niespójnych etykiet,
- analizę TWF, HDF, PWF, OSF,
- wykresy histogramów, boxplotów, pairplotów i scatterplotów.

### `notebooks/02_data_cleaning.ipynb`

Notebook dokumentujący czyszczenie danych.

Zawiera między innymi:

- dodanie `Temperature difference`,
- dodanie `Power [W]`,
- analizę rekordów `Machine failure == 1` bez przypisanego typu awarii,
- korektę tych rekordów,
- usunięcie kolumn ryzykownych dla modelowania,
- one-hot encoding kolumny `Type`.

### `notebooks/03_data_preparation.ipynb`

Notebook przygotowujący dane do ML.

Zawiera między innymi:

- wczytanie `data/processed/produkcja_clean.csv`,
- wydzielenie `X` i `y`,
- sprawdzenie rozkładu klas,
- `train_test_split` ze stratyfikacją,
- `StandardScaler` dla cech numerycznych,
- zapis `X_train_scaled.csv`, `X_test_scaled.csv`, `y_train.csv`, `y_test.csv`.

Uwaga: w aktualnym repo `03_data_preparation.ipynb` może mieć wyczyszczone outputy, ale jego kod zapisuje dane potrzebne do modelowania.

### `notebooks/04_modeling.ipynb`

Notebook z pierwszym porównaniem modeli bazowych.

Zawiera między innymi:

- wczytanie przygotowanych plików train/test,
- DummyClassifier jako baseline,
- Logistic Regression z `class_weight='balanced'`,
- Random Forest z `class_weight='balanced'`,
- Gradient Boosting,
- tabelę metryk,
- macierze pomyłek,
- analizę false negatives,
- analizę rekordów przeoczonych przez 1, 2 albo 3 modele,
- powrót do surowych danych w celu interpretacji trudnych przypadków,
- robocze grupy mechanizmów błędów.

---

## 12. Standard pracy, żeby AI widziało wyniki

Jeżeli zmieniasz kod, wystarczy zwykły commit i push:

```bash
git status
git add .
git commit -m "Opis zmiany"
git push
```

Jeżeli zmieniasz notebook i chcesz, by AI widziało wyniki komórek:

```text
1. Uruchom komórki w Jupyter / VS Code.
2. Upewnij się, że notebook pokazuje wyniki.
3. Zapisz notebook.
4. Sprawdź w git diff, czy w pliku .ipynb pojawiły się outputs.
5. Zrób commit.
6. Zrób git push.
```

Kontrola lokalna:

```bash
git diff notebooks/04_modeling.ipynb
```

Szukaj w diffie fragmentów typu:

```text
"execution_count": 1
"outputs": [
"text/plain"
"image/png"
```

Jeżeli nadal widzisz tylko:

```text
"execution_count": null
"outputs": []
```

to wyniki nie zostały zapisane w notebooku.

---

## 13. Lepszy sposób zachowywania ważnych wyników niż outputy notebooka

Outputy notebooka są wygodne, ale kruche. Można je przypadkiem usunąć przez `Clear All Outputs`.

Stabilniejsze miejsca na wyniki:

```text
PROJECT_CONTEXT.md       — najważniejsze decyzje i wnioski dla AI
reports/eda_summary.md   — opisowe wnioski z EDA
figures/                 — wykresy zapisane jako PNG
results/                 — metryki modeli, confusion matrix, raporty klasyfikacji
```

Rekomendacja:

```text
Najważniejsze wnioski wpisywać do Markdowna, a wykresy i tabele wyników zapisywać jako pliki.
Nie traktować .ipynb outputs jako jedynego źródła prawdy.
```

Przykład zapisu wykresu:

```python
plt.savefig("../figures/hdf_temperature_difference.png", dpi=300, bbox_inches="tight")
```

Przykład zapisu metryk:

```python
results_df.to_csv("../results/model_results_baseline.csv", index=False)
```

---

## 14. Zalecane następne kroki

Najbliższe dobre kroki w projekcie:

1. Zapisać wyniki modeli do stabilnego pliku, np. `results/model_results_baseline.csv`.
2. Dodać folder `results/` na metryki, confusion matrix i raporty klasyfikacji.
3. Dodać folder `reports/` na opisowe wnioski z EDA i modelowania.
4. Rozważyć dodanie cech interakcyjnych:

```text
Kryterium_OSF = Tool wear [min] * Torque [Nm]
```

5. Rozważyć dodatkowe cechy/reguły inspirowane mechaniką awarii:

```text
Power [W]
Temperature difference
Tool wear [min] * Torque [Nm]
```

6. Zrobić drugi eksperyment modelowania z dodatkowymi cechami i porównać false negatives.
7. Zrobić threshold tuning dla Random Forest i Gradient Boosting, szczególnie pod recall klasy 1.
8. Dodać interpretację cech: feature importance, permutation importance albo SHAP.
9. Rozbudować README o cel projektu, strukturę repo, dane, sposób uruchomienia i najważniejsze wyniki.
10. Dodać `requirements.txt` z bibliotekami projektu.

Proponowany kolejny notebook:

```text
notebooks/05_feature_engineering_and_thresholds.ipynb
```

albo skrypt:

```text
python/modeling.py
```

---

## 15. Jak AI powinno pracować z tym projektem

Przy kolejnych analizach AI powinno najpierw sprawdzić:

```text
PROJECT_CONTEXT.md
python/cleaning.py
notebooks/02_data_cleaning.ipynb
notebooks/03_data_preparation.ipynb
notebooks/04_modeling.ipynb
data/processed/produkcja_clean.csv
data/processed/X_train_scaled.csv
data/processed/X_test_scaled.csv
data/processed/y_train.csv
data/processed/y_test.csv
```

Następnie powinno ustalić, czy pytanie dotyczy:

```text
EDA
czyszczenia danych
przygotowania X/y
trenowania modelu
interpretacji wyników
false negatives
feature engineering
threshold tuning
organizacji repozytorium
Git/GitHub
```

Najważniejsze zasady:

```text
Nie zakładać, że wyniki komórek notebooka są widoczne, jeśli w pliku .ipynb nie ma zapisanych outputs.
Jeżeli potrzebne są dokładne liczby z EDA/modelowania, najpierw sprawdzić PROJECT_CONTEXT.md.
Jeżeli PROJECT_CONTEXT.md nie wystarcza, policzyć ponownie z CSV albo poprosić użytkownika o przesłanie aktualnego notebooka z outputami.
Nie używać TWF/HDF/PWF/OSF/RNF jako cech wejściowych do modelu binarnego.
Nie oceniać modelu wyłącznie po accuracy.
Przy awariach szczególnie pilnować false negatives i recall klasy 1.
```
