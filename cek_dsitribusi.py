import pandas as pd

df=pd.read_csv(
    "dataset_split_info_grouped.csv"
)

hasil=pd.crosstab(
    df['label'],
    df['set']
)

print(hasil)