#prepare_dataset.py
import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

print("="*60)
print("PREPARASI DATASET SKRIPSI - MOBILENETV2")
print("="*60)

# ==================== KONFIGURASI ====================
SOURCE_DIR = 'new_dataset_raw'  
TARGET_DIR = 'dataset'           # Folder tujuan
CLASS_NAMES = ['Dry', 'Normal', 'Oily', 'Sensitive', 'Combination']  
CLASS_MAPPING = {                 
    'Dry': 'dry',
    'Normal': 'normal',
    'Oily': 'oily',
    'Sensitive': 'sensitive',
    'Combination': 'combination' 
}

# ==================== CEK FOLDER ====================
print("\n[1] Memeriksa folder dataset baru...")

all_files = []
all_labels = []
class_counts = {}

for class_name in CLASS_NAMES:
    folder_path = os.path.join(SOURCE_DIR, class_name)
    mapped_name = CLASS_MAPPING[class_name]
    
    if os.path.exists(folder_path):
        # Ambil semua file gambar (jpg, jpeg, png)
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = len(images)
        class_counts[mapped_name] = count
        print(f"✅ {class_name} ({mapped_name}): {count} images")
        
        # Kumpulkan semua file
        for img_file in images:
            all_files.append(os.path.join(folder_path, img_file))
            all_labels.append(mapped_name)
    else:
        print(f"❌ {class_name}: folder tidak ditemukan!")

print(f"\n📊 Total images: {len(all_files)}")
print("\nDistribusi kelas:")
for class_name, count in class_counts.items():
    print(f"  {class_name}: {count}")

# ==================== SPLIT DATA ====================
print("\n[2] Melakukan split data...")

# Split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    all_files, all_labels, 
    test_size=0.3, 
    random_state=42, 
    stratify=all_labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print(f"\nHasil split:")
print(f"  - Train: {len(X_train)} images ({len(X_train)/len(all_files)*100:.1f}%)")
print(f"  - Validation: {len(X_val)} images ({len(X_val)/len(all_files)*100:.1f}%)")
print(f"  - Test: {len(X_test)} images ({len(X_test)/len(all_files)*100:.1f}%)")

# ==================== BUAT FOLDER TUJUAN ====================
print("\n[3] Membuat folder struktur...")

# Hapus folder target jika sudah ada
if os.path.exists(TARGET_DIR):
    print(f"  Menghapus folder {TARGET_DIR} yang lama...")
    shutil.rmtree(TARGET_DIR)

# Buat folder baru untuk 5 kelas
for dir_path in [os.path.join(TARGET_DIR, 'train'), 
                 os.path.join(TARGET_DIR, 'val'), 
                 os.path.join(TARGET_DIR, 'test')]:
    for class_name in ['dry', 'normal', 'oily', 'sensitive', 'combination']:
        os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)
        print(f"  Created: {os.path.join(dir_path, class_name)}")

# ==================== COPY FILES ====================
print("\n[4] Menyalin file gambar ke folder dataset...")

def copy_files(file_list, label_list, destination, set_name):
    success = 0
    total = len(file_list)
    
    print(f"\n  Memproses {set_name} ({total} files):")
    
    for i, (src_path, label) in enumerate(zip(file_list, label_list)):
        # Tampilkan progress setiap 50 file
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{total} files")
        
        # Dapatkan nama file
        filename = os.path.basename(src_path)
        
        # Tujuan
        dst = os.path.join(destination, label, filename)
        
        # Copy file
        shutil.copy2(src_path, dst)
        success += 1
    
    print(f"  {set_name} selesai: {success} berhasil")
    return success

print("\n" + "="*60)
train_success = copy_files(X_train, y_train, os.path.join(TARGET_DIR, 'train'), "TRAIN")
val_success = copy_files(X_val, y_val, os.path.join(TARGET_DIR, 'val'), "VALIDATION")
test_success = copy_files(X_test, y_test, os.path.join(TARGET_DIR, 'test'), "TEST")

# ==================== VERIFIKASI ====================
print("\n" + "="*60)
print("[5] VERIFIKASI HASIL AKHIR")
print("="*60)

print("\nIsi folder dataset/:")
total_keseluruhan = 0
for set_name in ['train', 'val', 'test']:
    print(f"\n{set_name.upper()}:")
    total_files = 0
    for class_name in ['dry', 'normal', 'oily', 'sensitive', 'combination']:
        path = os.path.join(TARGET_DIR, set_name, class_name)
        if os.path.exists(path):
            count = len(os.listdir(path))
            print(f"  - {class_name}: {count} files")
            total_files += count
            total_keseluruhan += count
    print(f"  TOTAL: {total_files} files")

# ==================== SIMPAN INFO ====================
print("\n[6] Menyimpan informasi split ke file...")
split_info = pd.DataFrame({
    'set': ['train'] * len(X_train) + ['val'] * len(X_val) + ['test'] * len(X_test),
    'file': [os.path.basename(f) for f in X_train + X_val + X_test],
    'label': y_train + y_val + y_test,
    'path': X_train + X_val + X_test
})
split_info.to_csv('dataset_split_info.csv', index=False)
print("  Split info saved to: dataset_split_info.csv")

print("\n" + "="*60)
print("✅ PREPARASI DATASET SELESAI!")
print("="*60)
print(f"\n📁 Dataset baru siap di: {os.path.abspath(TARGET_DIR)}")
print(f"📊 Total files: {total_keseluruhan}")