import os
import shutil
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

print("="*60)
print("PREPARE DATASET GROUPED")
print("="*60)

SOURCE_DIR='new_dataset_raw'
TARGET_DIR='dataset'

CLASS_NAMES=[
    'Dry',
    'Normal',
    'Oily',
    'Sensitive',
    'Combination'
]

CLASS_MAPPING={

    'Dry':'dry',
    'Normal':'normal',
    'Oily':'oily',
    'Sensitive':'sensitive',
    'Combination':'combination'

}

all_files=[]
all_labels=[]
groups=[]

print("\n[1] Membaca dataset...")

for class_name in CLASS_NAMES:

    folder=os.path.join(
        SOURCE_DIR,
        class_name
    )

    mapped=CLASS_MAPPING[class_name]

    files=[

        f for f in os.listdir(folder)

        if f.lower().endswith(
            ('.jpg','.jpeg','.png')
        )

    ]

    print(
        f"{class_name}: {len(files)}"
    )

    for f in files:

        path=os.path.join(
            folder,
            f
        )

        all_files.append(path)

        all_labels.append(
            mapped
        )

        # ===================
        # AMBIL NAMA DASAR
        # ===================

        name=os.path.splitext(
            f
        )[0]

        base=name

        patterns=[

            '_resized',
            '_rotated_90',
            '_rotated_180',
            '_flipped_horizontal',
            '_flipped_vertical'

        ]

        for p in patterns:

            base=base.replace(
                p,
                ''
            )

        base=base.replace(
            ' - Copy',
            ''
        )

        groups.append(base)

print(
    f"\nTotal file awal: {len(all_files)}"
)

df=pd.DataFrame({

    'file':all_files,
    'label':all_labels,
    'group':groups

})

print(
    f"Total group unik: {df['group'].nunique()}"
)

# ===================
# HAPUS DATASET LAMA
# ===================

if os.path.exists(
    TARGET_DIR
):

    print(
        "\nMenghapus dataset lama..."
    )

    shutil.rmtree(
        TARGET_DIR
    )

# ===================
# SPLIT TRAIN
# ===================

print(
    "\n[2] Group splitting..."
)

gss=GroupShuffleSplit(

    n_splits=1,
    test_size=0.3,
    random_state=42

)

train_idx,temp_idx=next(

    gss.split(

        df,
        groups=df['group']

    )

)

train_df=df.iloc[
    train_idx
]

temp_df=df.iloc[
    temp_idx
]

# ===================
# SPLIT VAL TEST
# ===================

gss2=GroupShuffleSplit(

    n_splits=1,
    test_size=0.5,
    random_state=42

)

val_idx,test_idx=next(

    gss2.split(

        temp_df,
        groups=temp_df['group']

    )

)

val_df=temp_df.iloc[
    val_idx
]

test_df=temp_df.iloc[
    test_idx
]

print("\nJumlah file:")

print(
f"Train:{len(train_df)}"
)

print(
f"Val:{len(val_df)}"
)

print(
f"Test:{len(test_df)}"
)

print(
f"Total:{len(train_df)+len(val_df)+len(test_df)}"
)

# ===================
# COPY
# ===================

print(
"\n[3] Copy files..."
)

for split_name,data in [

('train',train_df),
('val',val_df),
('test',test_df)

]:

    for _,row in data.iterrows():

        dst=os.path.join(

            TARGET_DIR,
            split_name,
            row['label']

        )

        os.makedirs(

            dst,
            exist_ok=True

        )

        shutil.copy2(

            row['file'],

            os.path.join(

                dst,

                os.path.basename(
                    row['file']
                )

            )

        )

# ===================
# SAVE CSV
# ===================

final_df=pd.concat([

    train_df.assign(
        set='train'
    ),

    val_df.assign(
        set='val'
    ),

    test_df.assign(
        set='test'
    )

])

final_df.to_csv(

    'dataset_split_info_grouped.csv',
    index=False

)

print(
"\nCSV tersimpan:"
)

print(
"dataset_split_info_grouped.csv"
)

# ===================
# CEK AKHIR
# ===================

count=0

for root,dirs,files in os.walk(
    TARGET_DIR
):

    count+=len(files)

print(
f"\nTotal file akhir:{count}"
)

if count==1800:

    print(
        "\n✅ JUMLAH SESUAI"
    )

else:

    print(
        "\n❌ ADA FILE HILANG"
    )

print("\nSELESAI")