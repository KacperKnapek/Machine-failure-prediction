import pandas as pd


sales = pd.read_csv("sales.csv")

print(sales["product_group"].value_counts())
