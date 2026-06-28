import pandas as pd

df = pd.read_csv(
    r"C:\Users\kknap\Desktop\Projekty\Projekty_Git\my-first-project\data\produkcja.csv")

df["Type"] = None
df["Type"] = df["Product ID"].str[0]

df.insert(loc=4, column="Air temperature [C]",
          value=df["Air temperature [K]"] - 273.15)

df.insert(loc=6, column="Process temperature [C]",
          value=df["Process temperature [K]"] - 273.15)

srednie_parametry = df.groupby("Type")[
    ['Air temperature [C]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']].mean()

awarie_per_type = df.groupby("Type")["Machine failure"].sum()

sciezka_excel = r"C:\Users\kknap\Desktop\Projekty\Projekty_Git\my-first-project\excel\produkcja.xlsx"

with pd.ExcelWriter(sciezka_excel) as writer:
    df.to_excel(writer, sheet_name="Czyste dane", index=False)
    srednie_parametry.to_excel(writer, sheet_name="Podsumowanie Typów")


print(f"Plik Excel został zapisany w lokalizacji: {sciezka_excel}")
