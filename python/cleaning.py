import pandas as pd
import numpy as np

df = pd.read_csv('./data/raw/produkcja.csv')

power = (df['Rotational speed [rpm]'] *
         df['Torque [Nm]'] * (2 * np.pi / 60)).round(2)
temp_difference = (df['Process temperature [K]'] -
                   df['Air temperature [K]']).round(1)

df.insert(loc=5, column='Temperature difference', value=temp_difference)
df.insert(loc=8, column='Power [W]', value=power)

df.loc[df[((df['TWF'] == 0) & (df['HDF'] == 0) & (df['PWF'] == 0) & (df['OSF'] == 0) & (
    df['RNF'] == 0)) & (df['Machine failure'] == 1)].index, 'Machine failure'] = 0

df.drop(columns=['UDI', 'Product ID', 'Air temperature [K]',
        'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], inplace=True)

dummies = pd.get_dummies(df, columns=['Type'], drop_first=True, dtype=int)

df = df.drop(columns=['Type'])
df.insert(loc=0, column='Type_L', value=dummies['Type_L'])
df.insert(loc=1, column='Type_M', value=dummies['Type_M'])

df.to_csv('./data/processed/produkcja_clean.csv', index=False)
print("Pomyślnie stworzono plik")
