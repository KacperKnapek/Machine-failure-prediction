# PROJECT_CONTEXT.md

Ten plik jest stałym kontekstem projektu dla Kacpra oraz narzędzi AI pracujących z repozytorium. Ma ograniczyć konieczność ciągłego tłumaczenia, co znajduje się w projekcie, jaki jest cel analizy, jakie decyzje zostały już podjęte i jakie wyniki należy zachować nawet wtedy, gdy notebooki zostaną zapisane bez `outputs`.

Ostatnia aktualizacja kontekstu: 2026-07-05.

---

## 0. Najważniejsza informacja dla AI

Przy każdej kolejnej pracy z tym repozytorium najpierw przeczytaj ten plik, a dopiero potem zaglądaj do notebooków i skryptów.

Ważne rozróżnienie:

```text
GitHub connector może pokazywać notebooki .ipynb bez outputs, nawet jeżeli użytkownik lokalnie ma wyniki komórek.
Użytkownik przesłał lokalne snapshoty notebooków z outputami: 01_eda.ipynb, 02_data_cleaning.ipynb, 03_data_preparation.ipynb.
Najważniejsze wyniki z tych outputów są zapisane w tym pliku.
```

Dlatego ten plik ma być źródłem trwałego kontekstu, jeśli notebooki zostaną wyczyszczone przez `Clear All Outputs`.

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

Surowy plik danych znajduje się tutaj:

```text
data/raw/produkcja.csv
```

Oryginalne kolumny danych:

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

Znaczenie głównych kolumn:

- `UDI` — unikalny identyfikator rekordu.
- `Product ID` — identyfikator produktu / maszyny; koduje typ maszyny przez pierwszą literę.
- `Type` — typ maszyny: H, L, M.
- `Air temperature [K]` — temperatura powietrza w kelwinach.
- `Process temperature [K]` — temperatura procesu w kelwinach.
- `Rotational speed [rpm]` — prędkość obrotowa.
- `Torque [Nm]` — moment obrotowy.
- `Tool wear [min]` — zużycie narzędzia w minutach.
- `Machine failure` — główna zmienna docelowa.
- `TWF`, `HDF`, `PWF`, `OSF`, `RNF` — flagi typów awarii.

Typy awarii:

```text
TWF = Tool Wear Failure
HDF = Heat Dissipation Failure
PWF = Power Failure
OSF = Overstrain Failure
RNF = Random Failure
```

---

## 3. Obecna struktura projektu rozpoznana w repozytorium

Ważne pliki rozpoznane w repozytorium:

```text
README.md
.gitignore
PROJECT_CONTEXT.md

data/raw/produkcja.csv
data/processed/produkcja_clean.csv

notebooks/01_eda.ipynb
notebooks/02_data_cleaning.ipynb
notebooks/03_data_preparation.ipynb

python/cleaning.py
```

Uwaga techniczna: dostępne narzędzie GitHub może nie mieć wygodnej funkcji pełnego listowania katalogów. Ten spis należy aktualizować ręcznie, gdy pojawią się nowe istotne pliki.

---

## 4. Najważniejsze wyniki EDA z lokalnych outputów notebooków

Źródło tej sekcji: przesłane przez użytkownika lokalne notebooki z outputami.

### 4.1. Rozmiar i kompletność danych po dodaniu cech inżynierskich

W `01_eda.ipynb` po dodaniu `Temperature difference` i `Power [W]` zbiór ma:

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

Wynik `df['Type'].value_counts()`:

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

W `01_eda.ipynb`, przed korektą niespójnych rekordów:

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

Wynik `df.groupby('Type')[['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum()`:

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

Wynik dla rekordów, gdzie suma flag `TWF`, `HDF`, `PWF`, `OSF`, `RNF` jest większa lub równa 2:

```text
24 rekordy mają co najmniej dwa typy awarii jednocześnie.
```

Wynik dla co najmniej trzech typów awarii:

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

Wynik dla `RNF == 1`:

```text
RNF = 1 występuje w 19 rekordach.
```

Wynik dla rekordów, gdzie istnieje dowolna flaga awarii, ale `Machine failure == 0`:

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

W `01_eda.ipynb` przed czyszczeniem znaleziono:

```text
9 rekordów, gdzie Machine failure = 1, ale TWF = HDF = PWF = OSF = RNF = 0.
```

Po korekcie w `02_data_cleaning.ipynb` wynik sprawdzenia wynosi:

```text
0 takich rekordów
```

Parametry tych 9 rekordów przed korektą były w przybliżeniu:

```text
Temperature difference: mean 9.87, min 8.20, max 11.20
Rotational speed [rpm]: mean 1507.67, min 1363, max 1710
Torque [Nm]: mean 41.69, min 27.3, max 54.0
Tool wear [min]: mean 117.22, min 20, max 210
Power [W]: mean 6453.39, min 4888.63, max 7707.58
```

Interpretacja i decyzja:

```text
Te rekordy potraktowano jako niespójność etykiety / szum i ustawiono Machine failure = 0.
To jest założenie projektowe, a nie prawda absolutna.
Kontrargument: mogły to być realne awarie bez poprawnie oznaczonego typu.
```

### 4.10. Awaria przy Tool wear = 0

W EDA znaleziono 3 rekordy, gdzie:

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

W `02_data_cleaning.ipynb` wynik korelacji:

```text
Air temperature [K] vs Process temperature [K] = 0.876107
Air temperature [K] vs Temperature difference = -0.699583
Process temperature [K] vs Temperature difference = -0.268413
```

Interpretacja:

```text
Air temperature i Process temperature są silnie dodatnio skorelowane.
Temperature difference wnosi inną informację, bo jest różnicą między temperaturą procesu i powietrza.
Usunięcie Air temperature [K] po utworzeniu Temperature difference jest metodologicznie uzasadnione, ale trzeba pamiętać, że jest to decyzja modelowa, nie konieczność matematyczna.
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

6. Koduje `Type` przez one-hot encoding z `drop_first=True`.

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

Kolumny i typy:

```text
Type_L                       int64
Type_M                       int64
Process temperature [K]    float64
Temperature difference     float64
Rotational speed [rpm]       int64
Torque [Nm]                float64
Power [W]                  float64
Tool wear [min]              int64
Machine failure              int64
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

Interpretacja:

```text
Stratyfikacja działa poprawnie: udział awarii w train i test jest taki sam.
To jest ważne przy niezbalansowanych klasach.
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

Po `StandardScaler` na train średnie są ~0, a odchylenia standardowe ~1.

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

Uwaga metodologiczna:

```text
Scaler został fitowany na X_train, a potem zastosowany do X_train i X_test.
To jest poprawny schemat, bo nie ma przecieku informacji z testu do treningu.
```

---

## 8. Decyzje metodologiczne, które trzeba zachować

### 8.1. Data leakage

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

### 8.2. Identyfikatory

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

### 8.3. Klasy niezbalansowane

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

### 8.4. Krytyczne założenie o korekcie etykiet

Korekta 9 rekordów `Machine failure = 1` bez typu awarii jest decyzją projektową.

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

## 9. Notebooki

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

Notebook przygotowujący dane pod ML.

Zawiera między innymi:

- wczytanie `data/processed/produkcja_clean.csv`,
- wydzielenie `X` i `y`,
- sprawdzenie rozkładu klas,
- `train_test_split` ze stratyfikacją,
- `StandardScaler` dla cech numerycznych.

---

## 10. Aktualny stan modelowania

Na podstawie rozpoznanych plików repozytorium nie widać jeszcze kompletnego etapu trenowania modeli ML.

Planowany klasyczny przebieg:

```text
1. Wczytaj data/processed/produkcja_clean.csv
2. Oddziel target: y = Machine failure
3. Oddziel cechy: X = reszta kolumn
4. Podziel dane na train/test z stratify=y
5. Przeskaluj cechy numeryczne, fitując scaler tylko na train
6. Porównaj modele bazowe
7. Oceń confusion matrix, recall, precision, F1, ROC-AUC, PR-AUC
```

Sugerowane pierwsze modele:

```text
LogisticRegression
RandomForestClassifier
GradientBoostingClassifier
```

---

## 11. Standard pracy, żeby AI widziało wyniki

Jeżeli zmieniasz kod, wystarczy zwykły commit i push:

```bash
git status
git add .
git commit -m "Opis zmiany"
git push
```

Jeżeli zmieniasz notebook i chcesz, by AI widziało wyniki komórek:

```text
1. Uruchom komórki w Jupyter / VS Code
2. Upewnij się, że notebook pokazuje wyniki
3. Zapisz notebook
4. Sprawdź w git diff, czy w pliku .ipynb pojawiły się outputs
5. Zrób commit
6. Zrób git push
```

Kontrola lokalna:

```bash
git diff notebooks/01_eda.ipynb
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

## 12. Lepszy sposób zachowywania ważnych wyników niż outputy notebooka

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
Najważniejsze wnioski wpisywać do Markdowna, a wykresy zapisywać jako pliki.
Nie traktować .ipynb outputs jako jedynego źródła prawdy.
```

Przykład zapisu wykresu:

```python
plt.savefig("../figures/hdf_temperature_difference.png", dpi=300, bbox_inches="tight")
```

---

## 13. Zalecane następne kroki

Najbliższe dobre kroki w projekcie:

1. Dokończyć `notebooks/03_data_preparation.ipynb` albo przenieść przygotowanie danych do skryptu.
2. Dodać `requirements.txt` z bibliotekami projektu.
3. Dodać folder `reports/` na wnioski z EDA.
4. Dodać folder `figures/` na wykresy zapisywane z notebooków.
5. Dodać folder `results/` na metryki modeli.
6. Przygotować pierwsze porównanie modeli bazowych.
7. Rozbudować README o cel projektu, dane, strukturę repo i sposób uruchomienia.

Proponowany kolejny notebook:

```text
notebooks/04_modeling.ipynb
```

albo skrypt:

```text
python/modeling.py
```

---

## 14. Jak AI powinno pracować z tym projektem

Przy kolejnych analizach AI powinno najpierw sprawdzić:

```text
PROJECT_CONTEXT.md
python/cleaning.py
notebooks/02_data_cleaning.ipynb
notebooks/03_data_preparation.ipynb
data/processed/produkcja_clean.csv
```

Następnie powinno ustalić, czy pytanie dotyczy:

```text
EDA
czyszczenia danych
przygotowania X/y
trenowania modelu
interpretacji wyników
organizacji repozytorium
Git/GitHub
```

Najważniejsza zasada:

```text
Nie zakładać, że wyniki komórek notebooka są widoczne, jeśli w pliku .ipynb nie ma zapisanych outputs.
Jeżeli potrzebne są dokładne liczby z EDA, najpierw sprawdzić PROJECT_CONTEXT.md.
Jeżeli PROJECT_CONTEXT.md nie wystarcza, policzyć ponownie z CSV albo poprosić użytkownika o przesłanie aktualnego notebooka z outputami.
```
