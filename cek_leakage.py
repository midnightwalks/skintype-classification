import pandas as pd

df = pd.read_csv(
    "dataset_split_info_grouped.csv"
)

leak=[]

for g in df['group'].unique():

    subset=df[
        df['group']==g
    ]['set'].unique()

    if len(subset)>1:

        leak.append(
            (g,subset)
        )

print(
    "Jumlah group bocor:",
    len(leak)
)

if len(leak)==0:

    print(
        "AMAN"
    )

else:

    print(
        leak[:20]
    )