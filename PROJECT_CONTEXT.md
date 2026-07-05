# PROJECT_CONTEXT.md

Ten plik jest stałym kontekstem projektu dla Kacpra oraz narzędzi AI pracujących z repozytorium. Jego celem jest ograniczenie konieczności ciągłego tłumaczenia, co znajduje się w projekcie, jaki jest cel analizy i jakie decyzje zostały już podjęte.

Stan opisany na podstawie repozytorium GitHub: 2026-07-05.

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

Na dalszym etapie można analizować również typy awarii, ale aktualny kierunek projektu to klasyfikacja binarna, a nie pełna klasyfikacja wieloetykietowa typów awarii.

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

## 3. Pliki projektu rozpoznane w repozytorium

Aktualnie rozpoznane ważne pliki:

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

Uwaga: narzędzie GitHub używane przez AI może nie mieć wygodnego pełnego listowania katalogów. Ten spis należy aktualizować ręcznie, gdy pojawią się nowe istotne pliki.

---

## 4. Inżynieria cech

W projekcie tworzona jest cecha:

```text
Temperature difference = Process temperature [K] - Air temperature [K]
```

Sens tej cechy:

```text
Pokazuje, o ile cieplejszy jest proces od otoczenia.
```

To może być ważne dla awarii typu HDF, ponieważ problem z odprowadzaniem ciepła powinien objawiać się nietypową relacją temperatury procesu do temperatury powietrza.

Tworzona jest także cecha mocy:

```text
Power [W] = Rotational speed [rpm] * Torque [Nm] * (2 * pi / 60)
```

Znaczenie fizyczne:

```text
Moc mechaniczna = moment obrotowy * prędkość kątowa
```

Ponieważ prędkość w danych jest w `rpm`, trzeba ją przeliczyć na radiany na sekundę przez czynnik:

```text
2 * pi / 60
```

---

## 5. Czyszczenie danych

W `python/cleaning.py` wykonywane są następujące kroki:

1. Wczytanie danych z:

```text
./data/raw/produkcja.csv
```

2. Obliczenie `Power [W]`.

3. Obliczenie `Temperature difference`.

4. Korekta rekordów, w których:

```text
Machine failure == 1
```

a jednocześnie wszystkie flagi typów awarii są równe zero:

```text
TWF == 0
HDF == 0
PWF == 0
OSF == 0
RNF == 0
```

Takie rekordy są traktowane jako szum / niespójność etykiety i zmieniane na:

```text
Machine failure = 0
```

5. Usunięcie kolumn:

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

Uzasadnienie:

- `UDI` jest identyfikatorem technicznym, nie cechą fizyczną procesu.
- `Product ID` jest unikalne i może wprowadzać model w zapamiętywanie rekordów zamiast uczenia się zależności.
- `Air temperature [K]` jest częściowo zastąpione przez `Temperature difference` oraz pozostawioną `Process temperature [K]`.
- `TWF`, `HDF`, `PWF`, `OSF`, `RNF` nie powinny być wejściem do modelu przewidującego `Machine failure`, ponieważ są informacją o rodzaju awarii i prowadziłyby do data leakage.

6. Zakodowanie `Type` przez one-hot encoding:

```text
Type_L
Type_M
```

Przy `drop_first=True` typ bazowy jest pomijany. Dla typów `H`, `L`, `M` bazą interpretacyjną jest zwykle `H`, a kolumny `Type_L` i `Type_M` mówią, czy maszyna należy do tych typów względem typu bazowego.

7. Zapis oczyszczonych danych do:

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
- Dane są już częściowo przygotowane do modelowania, ale przed modelem nadal warto wykonać podział train/test i skalowanie cech numerycznych tam, gdzie model tego wymaga.

---

## 7. Notebooki

### `notebooks/01_eda.ipynb`

Notebook do eksploracyjnej analizy danych.

Zawiera między innymi:

- import `pandas`, `matplotlib`, `seaborn`, `numpy`,
- wczytanie `../data/raw/produkcja.csv`,
- funkcję pomocniczą `plot_box_hist`,
- początkową analizę struktury danych,
- komórki typu `df.head()`, `df.isnull()` i podobne.

Ważna uwaga: w wersji widocznej przez GitHub komórki są zapisane bez wyników, np. `execution_count: null` oraz `outputs: []`. To znaczy, że AI widzi kod, ale nie widzi lokalnie wygenerowanych tabel i wykresów, dopóki notebook nie zostanie uruchomiony, zapisany i wypchnięty przez `git push`.

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

Na moment utworzenia tego pliku kontekstowego notebook wygląda prawie pusty. Prawdopodobnie powinien służyć do etapu przygotowania danych pod ML, np.:

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
8. Oceń confusion matrix, recall, precision, F1, ROC-AUC
```

Sugerowane pierwsze modele:

```text
LogisticRegression
RandomForestClassifier
GradientBoostingClassifier
```

W tym projekcie sama accuracy może być myląca, ponieważ awarie są rzadkie. Ważniejsze metryki:

```text
recall dla klasy 1
precision dla klasy 1
F1-score dla klasy 1
confusion matrix
ROC-AUC
PR-AUC / Average Precision
```

---

## 9. Ważne decyzje metodologiczne

### Data leakage

Nie wolno używać `TWF`, `HDF`, `PWF`, `OSF`, `RNF` jako cech wejściowych do modelu przewidującego `Machine failure`, jeśli model ma działać predykcyjnie.

Powód:

```text
Te kolumny są niemal bezpośrednim opisem awarii, czyli zawierają informację, którą model ma dopiero przewidzieć.
```

To byłby klasyczny przeciek danych.

### Product ID

`Product ID` może być mylące dla modelu, bo jest identyfikatorem i może prowadzić do zapamiętywania próbek. W projekcie zostało usunięte.

### RNF

`RNF`, czyli Random Failure, jest szczególnie problematyczne, bo z definicji może być słabo przewidywalne na podstawie parametrów procesu. Przy interpretacji błędów modelu należy pamiętać, że część awarii może nie mieć silnego sygnału w cechach.

### Klasy niezbalansowane

Awarii jest znacznie mniej niż normalnych przypadków. Dlatego model może osiągać wysoką accuracy, przewidując prawie zawsze brak awarii. To byłby model pozornie dobry, ale praktycznie słaby.

---

## 10. Zalecany standard pracy z GitHubem

Aby AI widziało aktualny stan projektu, po istotnych zmianach warto wykonywać:

```bash
git status
git add .
git commit -m "Opis zmiany"
git push
```

Jeżeli zmieniasz notebook i chcesz, by AI widziało wyniki komórek:

```text
1. Uruchom komórki w Jupyter / VS Code
2. Zapisz notebook
3. Zrób commit
4. Zrób git push
```

Bez tego AI widzi kod notebooka, ale nie widzi lokalnych wyników, wykresów ani tabel.

---

## 11. Zalecane następne kroki

Najbliższe dobre kroki w projekcie:

1. Uporządkować `notebooks/03_data_preparation.ipynb`.
2. Dodać skrypt lub notebook modelujący, np.:

```text
notebooks/04_modeling.ipynb
```

albo:

```text
python/modeling.py
```

3. Dodać `requirements.txt` z bibliotekami projektu.
4. Dodać folder `reports/` na wnioski z EDA.
5. Dodać folder `figures/` na wykresy zapisywane z notebooków.
6. Przygotować pierwsze porównanie modeli bazowych.
7. Opisać w README cel projektu, dane, strukturę repo i sposób uruchomienia.

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

Następnie powinno ustalić:

```text
czy pytanie dotyczy EDA,
czy pytanie dotyczy czyszczenia danych,
czy pytanie dotyczy przygotowania X/y,
czy pytanie dotyczy trenowania modelu,
czy pytanie dotyczy interpretacji wyników.
```

Najważniejsza zasada:

```text
Nie zakładać, że wyniki komórek notebooka są widoczne, jeśli w pliku .ipynb nie ma zapisanych outputs.
```
