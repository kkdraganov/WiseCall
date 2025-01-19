import pandas as pd

df = pd.read_csv('data.csv')

X = df['TEXT']
for text in X:
    print(type(text))  # <class 'str'>