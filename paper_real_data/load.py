import pandas as pd

# lettura del file con all'interno le recensioni di oggetti recensiti più di 20000 volte
df = pd.read_csv('famous.csv', dtype={'asin': str}).sample(frac=1)
