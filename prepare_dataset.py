import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from collections import Counter
from PIL import Image

# ==============================================================================
# CONFIGURATION AND SETUP
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'dataset_v2')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

HAM10000_META = os.path.join(BASE_DIR, 'HAM10000_metadata.csv')
HAM_DIRS = [
    os.path.join(BASE_DIR, 'HAM10000_images_part_1'),
    os.path.join(BASE_DIR, 'HAM10000_images_part_2')
]

VITILIGO_BASE = os.path.join(BASE_DIR, 'healthy_temp', 'Dermatological Images for Vitiligo Disease Dataset')
HEALTHY_BASE = os.path.join(BASE_DIR, 'healthy_dataset')

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'healthy', 'mel', 'nv', 'vasc', 'vitiligo']
SPLITS = ['train', 'val', 'test']
HEALTHY_MAX_PER_SPLIT = 2000  # Cap healthy class per split to avoid massive imbalance

def setup_directories():
    """Create fresh dataset_v2 directory with train/val/test splits and class subdirectories."""
    print("[INFO] Setting up directories...")
    if os.path.exists(OUTPUT_DIR):
        print(f"[INFO] Removing existing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)
            
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("[INFO] Directories created successfully.")

# ==============================================================================
# DATA GATHERING
# ==============================================================================
def find_ham10000_image(image_id):
    """Find the full path for a given HAM10000 image_id."""
    filename = f"{image_id}.jpg"
    for d in HAM_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None

def gather_ham10000():
    """Process HAM10000 metadata and split by patient (lesion_id)."""
    print("[INFO] Processing HAM10000 metadata...")
    if not os.path.exists(HAM10000_META):
        print("[ERROR] HAM10000_metadata.csv not found!")
        return [], [], []
        
    df = pd.read_csv(HAM10000_META)
    
    # Check for missing paths
    df['image_path'] = df['image_id'].apply(find_ham10000_image)
    missing = df['image_path'].isna().sum()
    if missing > 0:
        print(f"[WARNING] {missing} HAM10000 images not found!")
        df = df.dropna(subset=['image_path'])
        
    # GroupShuffleSplit to prevent data leakage (patient level)
    # 70% Train, 30% Temp (which will be split 15% Val / 15% Test)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, groups=df['lesion_id']))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    
    # Split Temp into Val and Test (50/50 of the 30% -> 15% each)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df['lesion_id']))
    
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]
    
    def df_to_tuples(dataframe):
        return [(row['image_path'], row['dx']) for _, row in dataframe.iterrows()]
        
    print(f"[INFO] HAM10000 Split: {len(train_df)} Train, {len(val_df)} Val, {len(test_df)} Test")
    return df_to_tuples(train_df), df_to_tuples(val_df), df_to_tuples(test_df)

def gather_vitiligo():
    """Gather all vitiligo images and resplit."""
    print("[INFO] Gathering Vitiligo images...")
    paths = []
    for split in ['train', 'valid', 'test']:
        d = os.path.join(VITILIGO_BASE, split, 'Vitiligo Disease')
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(d, f))
                    
    if not paths:
        print("[WARNING] No Vitiligo images found!")
        return [], [], []
        
    train_paths, temp_paths = train_test_split(paths, train_size=0.7, random_state=42)
    val_paths, test_paths = train_test_split(temp_paths, train_size=0.5, random_state=42)
    
    def to_tuples(p_list):
        return [(p, 'vitiligo') for p in p_list]
        
    return to_tuples(train_paths), to_tuples(val_paths), to_tuples(test_paths)

def gather_healthy():
    """Gather all healthy images and resplit."""
    print("[INFO] Gathering Healthy Skin images...")
    paths = []
    
    # 1. From healthy_dataset
    for split in ['train', 'test']:
        d = os.path.join(HEALTHY_BASE, split, 'healthy')
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(d, f))
                    
    # 2. From Vitiligo Dataset's Healthy Skin folder
    for split in ['train', 'valid', 'test']:
        d = os.path.join(VITILIGO_BASE, split, 'Healthy Skin')
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(d, f))
                    
    if not paths:
        print("[WARNING] No Healthy images found!")
        return [], [], []
        
    train_paths, temp_paths = train_test_split(paths, train_size=0.7, random_state=42)
    val_paths, test_paths = train_test_split(temp_paths, train_size=0.5, random_state=42)
    
    # Cap per split if necessary
    if len(train_paths) > HEALTHY_MAX_PER_SPLIT:
        train_paths = train_test_split(train_paths, train_size=HEALTHY_MAX_PER_SPLIT, random_state=42)[0]
    if len(val_paths) > HEALTHY_MAX_PER_SPLIT:
        val_paths = train_test_split(val_paths, train_size=HEALTHY_MAX_PER_SPLIT, random_state=42)[0]
    if len(test_paths) > HEALTHY_MAX_PER_SPLIT:
        test_paths = train_test_split(test_paths, train_size=HEALTHY_MAX_PER_SPLIT, random_state=42)[0]
        
    def to_tuples(p_list):
        return [(p, 'healthy') for p in p_list]
        
    return to_tuples(train_paths), to_tuples(val_paths), to_tuples(test_paths)

# ==============================================================================
# COPYING AND REPORTING
# ==============================================================================
def is_valid_image(path):
    """Check if file is a valid image to prevent corruption errors later."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def copy_files(split_name, data_tuples):
    """Copy files to the output directory structure, verifying integrity."""
    print(f"[INFO] Copying {split_name} files ({len(data_tuples)} items)...")
    copied = 0
    skipped = 0
    
    for src, cls in data_tuples:
        if not is_valid_image(src):
            skipped += 1
            continue
            
        filename = os.path.basename(src)
        # Ensure unique filenames in case of duplicates from different folders
        dest = os.path.join(OUTPUT_DIR, split_name, cls, filename)
        if os.path.exists(dest):
            name, ext = os.path.splitext(filename)
            dest = os.path.join(OUTPUT_DIR, split_name, cls, f"{name}_copy{ext}")
            
        shutil.copy2(src, dest)
        copied += 1
        
    print(f"[INFO] Copied: {copied}, Skipped (corrupt/invalid): {skipped}")
    
def generate_report(train_data, val_data, test_data):
    """Generate and save detailed dataset statistics."""
    report_path = os.path.join(REPORTS_DIR, 'dataset_report.txt')
    print("[INFO] Generating dataset statistics report...")
    
    train_counts = Counter([cls for _, cls in train_data])
    val_counts = Counter([cls for _, cls in val_data])
    test_counts = Counter([cls for _, cls in test_data])
    
    total_counts = Counter()
    total_counts.update(train_counts)
    total_counts.update(val_counts)
    total_counts.update(test_counts)
    
    total_images = sum(total_counts.values())
    
    lines = []
    lines.append("======================================================")
    lines.append("              DATASET STATISTICS REPORT               ")
    lines.append("======================================================")
    lines.append(f"Total Images: {total_images}\n")
    
    lines.append(f"{'Class':<15} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    lines.append("-" * 55)
    
    for cls in CLASSES:
        tr = train_counts.get(cls, 0)
        v = val_counts.get(cls, 0)
        te = test_counts.get(cls, 0)
        tot = tr + v + te
        lines.append(f"{cls:<15} | {tr:<8} | {v:<8} | {te:<8} | {tot:<8}")
        
    lines.append("-" * 55)
    lines.append(f"{'TOTAL':<15} | {len(train_data):<8} | {len(val_data):<8} | {len(test_data):<8} | {total_images:<8}\n")
    
    lines.append("Class Imbalance (Total distribution):")
    for cls in CLASSES:
        tot = total_counts.get(cls, 0)
        pct = (tot / total_images * 100) if total_images > 0 else 0
        lines.append(f"  {cls:<12}: {pct:.2f}%")
        
    report_content = "\n".join(lines)
    print("\n" + report_content + "\n")
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    print(f"[INFO] Report saved to {report_path}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("[INFO] Starting Dataset Preparation Process...")
    
    setup_directories()
    
    ham_train, ham_val, ham_test = gather_ham10000()
    vit_train, vit_val, vit_test = gather_vitiligo()
    hea_train, hea_val, hea_test = gather_healthy()
    
    train_data = ham_train + vit_train + hea_train
    val_data = ham_val + vit_val + hea_val
    test_data = ham_test + vit_test + hea_test
    
    copy_files('train', train_data)
    copy_files('val', val_data)
    copy_files('test', test_data)
    
    generate_report(train_data, val_data, test_data)
    
    print("[INFO] Dataset Preparation Completed Successfully.")