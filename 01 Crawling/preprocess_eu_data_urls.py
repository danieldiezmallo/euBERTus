import pandas as pd
import numpy as np

df = pd.read_csv('data/00_raw/Non-exhaustive list of COVID-19 related document on EUR-Lex.csv')

df['html_to_download'] = df.html_to_download.apply(lambda x: '/'.join(x.split('/')[:-1]))
df['id'] = df.index + 1
df = df.where(df.notnull(), None)

df = df[["id", "title_", "html_to_download", "authors", "date_document", "celex", "Full_OJ"]]
print(df)
df.to_csv('data/00_raw/extracted/eu_data_preprocessed.csv', index=False)