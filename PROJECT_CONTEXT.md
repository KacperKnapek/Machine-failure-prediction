# PROJECT_CONTEXT.md

Ten plik jest stałym kontekstem projektu dla Kacpra oraz narzędzi AI pracujących z repozytorium. Ma ograniczyć konieczność ciągłego tłumaczenia, co znajduje się w projekcie, jaki jest cel analizy, jakie decyzje zostały już podjęte i jakie wyniki należy zachować nawet wtedy, gdy notebooki zostaną zapisane bez `outputs`.

Ostatnia aktualizacja kontekstu: 2026-07-05.

---

## 0. Najważniejsza informacja dla AI

Przy każdej kolejnej pracy z tym repozytorium najpierw przeczytaj ten plik, a dopiero potem zaglądaj do notebooków i skryptów.

Aktualny stan po sprawdzeniu repozytorium:

```text
Notebooki .ipynb są widoczne przez GitHub, ale ich wyniki komórek nie są aktualnie zapisane.
W plikach notebooków widoczne są execution_count: null oraz outputs: [].
```

To znaczy:

```text
AI widzi kod i markdown notebooków, ale nie widzi tabel, wykresów ani tekstowych wyników komórek,
jeżeli te wyniki nie zostały zapisane w samym pliku .ipynb i wypchnięte przez git push.
```

Dlatego najważniejsze stabilne wyniki i decyzje projektowe są zapisane tutaj, a nie wyłącznie w outputach notebooków.

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
- `Product ID` — identyfikator produktu / maszyny; może kodować typ maszyny.
- `Type` — typ maszyny, np. L, M, H.
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

## 4. Wyniki i decyzje, które mają przetrwać czyszczenie outputów notebooka

### 4.1. Decyzja o targetcie

Target projektu:

```text
Machine failure
```

Charakter zadania:

```text
binary classification
```

Model ma przewidywać sam fakt awarii, niekoniecznie jej dokładny typ.

### 4.2. Decyzja o usunięciu identyfikatorów

Z dotychczasowej analizy projektu:

```text
UDI i Product ID nie powinny być cechami wejściowymi modelu.
```

Uzasadnienie:

- `UDI` jest identyfikatorem technicznym, a nie zjawiskiem fizycznym.
- `Product ID` ma bardzo wysoką unikalność i może prowadzić do zapamiętywania rekordów.
- Wcześniejsze sprawdzenie projektu wskazywało, że `Product ID` jednoznacznie koduje `Type`, więc jest redundantne względem typu maszyny.

### 4.3. Decyzja o usunięciu flag typów awarii z cech wejściowych

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

### 4.4. Decyzja o cechach pochodnych

Tworzone są dwie ważne cechy inżynierskie:

```text
Temperature difference = Process temperature [K] - Air temperature [K]
Power [W] = Rotational speed [rpm] * Torque [Nm] * (2 * pi / 60)
```

Interpretacja:

- `Temperature difference` mówi, o ile proces jest cieplejszy od otoczenia.
- `Power [W]` przybliża moc mechaniczną układu.

Fizycznie:

```text
Moc mechaniczna = moment obrotowy * prędkość kątowa
```

Ponieważ prędkość w danych jest w `rpm`, przelicznik do rad/s wynosi:

```text
2 * pi / 60
```

### 4.5. Decyzja o korekcie niespójnych rekordów awarii

W czyszczeniu danych korygowane są rekordy spełniające warunek:

```text
Machine failure == 1
oraz
TWF == 0, HDF == 0, PWF == 0, OSF == 0, RNF == 0
```

Decyzja projektowa:

```text
Takie rekordy są traktowane jako szum / niespójność etykiety i ustawiane na Machine failure = 0.
```

Uwaga krytyczna:

To jest założenie metodologiczne, nie prawda absolutna. Możliwy kontrargument: awaria mogła istnieć, ale typ awarii nie został poprawnie opisany. W tym projekcie przyjęto jednak interpretację, że brak jakiejkolwiek flagi typu awarii oznacza brak wiarygodnego potwierdzenia awarii.

### 4.6. RNF jako problem interpretacyjny

`RNF`, czyli `Random Failure`, jest szczególnie problematyczne.

Interpretacja praktyczna:

```text
Nie każda awaria musi być dobrze przewidywalna z dostępnych parametrów procesu.
```

Jeżeli model myli część przypadków związanych z RNF, nie musi to oznaczać, że model jest źle zbudowany. Może oznaczać, że w danych nie ma wystarczającego sygnału przyczynowego.

### 4.7. Problem niezbalansowanych klas

Awarii jest znacznie mniej niż normalnych przypadków. Dlatego sama `accuracy` może być myląca.

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

---

## 5. Czyszczenie danych w `python/cleaning.py`

Skrypt `python/cleaning.py` wykonuje następujący proces:

1. Wczytuje dane z:

```text
./data/raw/produkcja.csv
```

2. Oblicza `Power [W]`.

3. Oblicza `Temperature difference`.

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

Rozpoznane kolumny po czyszczeniu:

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

- `Machine failure` pozostaje targetem.
- Pozostałe kolumny są kandydatami na cechy wejściowe modelu.
- Dane są częściowo przygotowane do modelowania.
- Przed modelem nadal należy wykonać podział train/test i ewentualne skalowanie cech numerycznych.

---

## 7. Notebooki

### `notebooks/01_eda.ipynb`

Notebook do eksploracyjnej analizy danych.

Zawiera między innymi:

- import `pandas`, `matplotlib`, `seaborn`, `numpy`,
- wczytanie `../data/raw/produkcja.csv`,
- funkcję pomocniczą `plot_box_hist`,
- analizę struktury danych,
- komórki typu `df.head()`, `df.isnull()`, `df.shape`, `df.dtypes`, `df.describe()`, `df.duplicated().sum()`.

Aktualny stan zapisu notebooka:

```text
execution_count: null
outputs: []
```

Czyli widoczny jest kod, ale nie są widoczne wyniki tabelaryczne ani wykresy.

### `notebooks/02_data_cleaning.ipynb`

Notebook dokumentujący czyszczenie danych.

Zawiera między innymi:

- dodanie `Temperature difference`,
- dodanie `Power [W]`,
- analizę rekordów `Machine failure == 1` bez przypisanego typu awarii,
- korektę tych rekordów,
- usunięcie kolumn ryzykownych dla modelowania,
- one-hot encoding kolumny `Type`.

Aktualny stan zapisu notebooka:

```text
execution_count: null
outputs: []
```

Czyli widoczny jest kod i decyzje, ale nie wyniki wykonania komórek.

### `notebooks/03_data_preparation.ipynb`

Na moment aktualizacji kontekstu notebook jest praktycznie pusty.

Powinien docelowo służyć do etapu przygotowania danych pod ML, np.:

- wydzielenie `X` i `y`,
- `train_test_split`,
- `stratify=y`,
- skalowanie cech numerycznych,
- przygotowanie pipeline'u,
- zapis gotowych zbiorów lub przejście do modelowania.

---

## 8. Aktualny stan modelowania

Na podstawie rozpoznanych plików repozytorium nie widać jeszcze kompletnego etapu trenowania modeli ML.

Planowany klasyczny przebieg:

```text
1. Wczytaj data/processed/produkcja_clean.csv
2. Oddziel target: y = Machine failure
3. Oddziel cechy: X = reszta kolumn
4. Podziel dane na train/test
5. Zastosuj stratify=y, bo awarie są rzadkie
6. Przeskaluj cechy numeryczne dla modeli wrażliwych na skalę
7. Porównaj modele bazowe
8. Oceń confusion matrix, recall, precision, F1, ROC-AUC, PR-AUC
```

Sugerowane pierwsze modele:

```text
LogisticRegression
RandomForestClassifier
GradientBoostingClassifier
```

---

## 9. Standard pracy, żeby AI widziało wyniki

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

## 10. Lepszy sposób zachowywania ważnych wyników niż outputy notebooka

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

## 11. Zalecane następne kroki

Najbliższe dobre kroki w projekcie:

1. Uporządkować `notebooks/03_data_preparation.ipynb`.
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

## 12. Jak AI powinno pracować z tym projektem

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
```

Jeżeli potrzebne są dokładne liczby z EDA, a notebook nie ma outputs, należy je ponownie policzyć z pliku CSV albo poprosić użytkownika o zapisanie wyników do `reports/eda_summary.md`.
