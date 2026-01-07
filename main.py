"""
Sustav za prepoznavanje hoda - Lokalna verzija
Autor: Dominik Černjević, Domagoj Gerić
Datum: 2026-01-06
"""
from multiprocessing import Pool, cpu_count
import functools

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix, 
    roc_curve, 
    auc
)

# ================== KONFIGURACIJA ==================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, 'dataset')
RESULTS_PATH = os.path.join(BASE_PATH, 'results')
IMG_SIZE = (256, 256)

os.makedirs(RESULTS_PATH, exist_ok=True)

print("="*60)
print(f"Putanja do dataseta: {DATASET_PATH}")
print(f"Rezultati će biti spremljeni u: {RESULTS_PATH}")
print("="*60)

def process_single_sequence(args):
    subject, seq, angle, root_folder = args
    
    angle_path = os.path.join(root_folder, subject, seq, angle)
    
    if not os.path.isdir(angle_path):
        return None, None
    
    image_names = sorted([f for f in os.listdir(angle_path) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))])
    image_paths = [os.path.join(angle_path, img) for img in image_names]
    
    if len(image_paths) > 5:
        gei = generate_gei(image_paths)
        if gei is not None:
            return gei.flatten(), subject
    
    return None, None


def load_dataset_parallel(root_folder, target_angle='090'):
    print(f"\nParalelno učitavanje (koristi {cpu_count()} jezgri)...")
    
    subjects = sorted([d for d in os.listdir(root_folder) 
                      if os.path.isdir(os.path.join(root_folder, d))])
    
    # Pripremi sve zadatke
    tasks = []
    for subject in subjects:
        subject_path = os.path.join(root_folder, subject)
        sequences = os.listdir(subject_path)
        
        for seq in sequences:
            seq_path = os.path.join(subject_path, seq)
            if not os.path.isdir(seq_path):
                continue
            
            angles = os.listdir(seq_path)
            for angle in angles:
                if target_angle is not None and angle != target_angle:
                    continue
                
                tasks.append((subject, seq, angle, root_folder))
    
    print(f"   Ukupno zadataka: {len(tasks)}")
    
    # Paralelno izvršavanje
    X, y = [], []
    
    with Pool(cpu_count()) as pool:
        results = pool.map(process_single_sequence, tasks)
    
    # Sakupi rezultate
    for gei, label in results:
        if gei is not None:
            X.append(gei)
            y.append(label)
    
    print(f"Učitano {len(X)} GEI uzoraka")
    return np.array(X), np.array(y)

# Obrada slike: binarizacija i centriranje
def preprocess_frame(img, img_size):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    if h < 10 or w < 5: 
        return None

    roi = thresh[y:y+h, x:x+w]

    roi_resized = cv2.resize(roi, img_size, interpolation=cv2.INTER_AREA)
    
    return roi_resized

def generate_gei(image_paths, debug=False):
    if not image_paths:
        return None

    gei_accumulator = None
    count = 0
    
    for i, path in enumerate(image_paths):
        img = cv2.imread(path, 0) # Učitaj kao grayscale
        
        if img is None:
            continue
            
        # --- OVDJE JE PROMJENA: Preprocess (crop & binarize) ---
        processed_img = preprocess_frame(img, IMG_SIZE)
        
        if processed_img is None:
            continue
            
        if gei_accumulator is None:
            gei_accumulator = np.zeros(processed_img.shape, dtype=np.float32)
        
        gei_accumulator += processed_img
        count += 1
    
    if count == 0 or gei_accumulator is None:
        if debug: print("Nije pronađena niti jedna validna silueta.")
        return None

    gei_img = gei_accumulator / count
    gei_img = ((gei_img - gei_img.min()) / (gei_img.max() - gei_img.min() + 1e-5)) * 255
    
    return gei_img.astype(np.uint8)

def load_condition_data(root_folder, condition_prefix, target_angle='090'):
    X_cond, y_cond = [], []
    
    subjects = sorted([d for d in os.listdir(root_folder) 
                      if os.path.isdir(os.path.join(root_folder, d))])
    
    total = len(subjects)
    processed = 0
    
    for subject in subjects:
        subject_path = os.path.join(root_folder, subject)
        if not os.path.isdir(subject_path): 
            continue
        
        sequences = os.listdir(subject_path)
        for seq in sequences:
            if not seq.startswith(condition_prefix):
                continue
            
            seq_path = os.path.join(subject_path, seq)
            if not os.path.isdir(seq_path): 
                continue
            
            angles = os.listdir(seq_path)
            
            for angle in angles:
                if target_angle is not None and angle != target_angle:
                    continue
                
                angle_path = os.path.join(seq_path, angle)
                if not os.path.isdir(angle_path):
                    continue
                
                image_names = sorted([f for f in os.listdir(angle_path) 
                                     if f.endswith(('.png', '.jpg', '.jpeg'))])
                image_paths = [os.path.join(angle_path, img) for img in image_names]
                
                if len(image_paths) > 5:
                    gei = generate_gei(image_paths)
                    if gei is not None:
                        X_cond.append(gei.flatten())
                        y_cond.append(subject)
        
        processed += 1
        if processed % 10 == 0 or processed == total:
            print(f".", end="", flush=True)
    
    return np.array(X_cond), np.array(y_cond)


# ================== GLAVNI PROGRAM ==================

def main():
    """
    Glavna funkcija - izvršava cijeli pipeline.
    """
    
    # 1. UČITAVANJE PODATAKA
    print("\n" + "="*60)
    print("FAZA 1: UČITAVANJE I PRIPREMA PODATAKA")
    print("="*60)
    
    X, y = load_dataset_parallel(DATASET_PATH)
    
    if len(X) == 0:
        print("Nema podataka! Provjerite putanju do dataseta.")
        return
    
    # 2. PODJELA PODATAKA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nPodjela podataka:")
    print(f"Trening set: {X_train.shape[0]} uzoraka")
    print(f"Test set: {X_test.shape[0]} uzoraka")
    print(f"Broj osoba: {len(np.unique(y))}")
    
    # 3. TRENIRANJE MODELA
    print("\n" + "="*60)
    print("FAZA 2: TRENIRANJE MODELA")
    print("="*60)
    
    n_components = min(len(X_train), 50)
    print(f"PCA komponente: {n_components}")
    
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components),
        SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    )
    
    print("Treniranje u tijeku...")
    model.fit(X_train, y_train)
    print("Model istreniran!")
    
    # 4. EVALUACIJA
    print("\n" + "="*60)
    print("FAZA 3: EVALUACIJA PERFORMANSI")
    print("="*60)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nUkupna točnost: {acc*100:.2f}%")
    print("\nIzvještaj klasifikacije:")
    print(classification_report(y_test, y_pred))
    
    # 5. CONFUSION MATRIX
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Matrica Zabune (Confusion Matrix)', fontsize=16)
    plt.ylabel('Stvarni Identitet', fontsize=12)
    plt.xlabel('Predviđeni Identitet', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'confusion_matrix.png'), dpi=300)
    print(f"Confusion matrix spremljena u: results/confusion_matrix.png")
    plt.show()
    
    # 6. GEI PRIMJER
    sample_gei = X_train[0].reshape(IMG_SIZE)
    plt.figure(figsize=(5, 5))
    plt.imshow(sample_gei, cmap='gray')
    plt.title("Primjer generirane GEI slike", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'gei_sample.png'), dpi=300)
    print(f"GEI primjer spremljen u: results/gei_sample.png")
    plt.show()
    
    # 7. FAR/FRR ANALIZA
    print("\n" + "="*60)
    print("FAZA 4: BIOMETRIJSKA ANALIZA (FAR, FRR, EER)")
    print("="*60)
    
    classes = np.unique(y)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_test_bin.ravel(), y_score.ravel())
    
    # Obrnuti redoslijed ako je potrebno
    if len(thresholds) > 1 and thresholds[0] > thresholds[-1]:
        fpr = fpr[::-1]
        tpr = tpr[::-1]
        thresholds = thresholds[::-1]
    
    far = fpr
    frr = 1 - tpr
    
    idx_eer = np.nanargmin(np.absolute(far - frr))
    eer_val = far[idx_eer]
    eer_threshold = thresholds[idx_eer]
    
    print(f"\n Equal Error Rate (EER): {eer_val*100:.2f}%")
    print(f"Optimalni prag: {eer_threshold:.4f}")
    print(f"AUC: {auc(fpr, tpr):.4f}")
    
    # Tablica pragova
    print("\n" + "="*60)
    print("Analiza različitih pragova:")
    print("="*60)
    print(f"{'Prag':<12} | {'FAR (%)':<10} | {'FRR (%)':<10}")
    print("-" * 40)
    
    for t in [0.3, 0.5, 0.7, 0.85, 0.95]:
        idx = (np.abs(thresholds - t)).argmin()
        print(f"{t:<12.2f} | {far[idx]*100:<10.2f} | {frr[idx]*100:<10.2f}")
    
    # FAR/FRR Graf
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, far, label='FAR (False Acceptance Rate)', color='red', linewidth=2)
    plt.plot(thresholds, frr, label='FRR (False Rejection Rate)', color='blue', linewidth=2)
    plt.plot(eer_threshold, eer_val, 'go', markersize=10, label=f'EER = {eer_val*100:.1f}%')
    plt.xlabel('Prag osjetljivosti (Threshold)', fontsize=12)
    plt.ylabel('Stopa pogreške (Error Rate)', fontsize=12)
    plt.title('Odnos FAR i FRR ovisno o pragu odluke', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'far_frr_curve.png'), dpi=300)
    print(f"\nFAR/FRR krivulja spremljena u: results/far_frr_curve.png")
    plt.show()
    
    # ROC Krivulja
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC krivulja (AUC = {auc(fpr, tpr):.3f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Nasumično pogađanje')
    plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
    plt.ylabel('True Acceptance Rate (1 - FRR)', fontsize=12)
    plt.title('ROC Krivulja', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'roc_curve.png'), dpi=300)
    print(f"ROC krivulja spremljena u: results/roc_curve.png")
    plt.show()

    # 8. CROSS-CONDITION TESTIRANJE (Optimizirano)
    print("\n" + "="*60)
    print("FAZA 5: TESTIRANJE ROBUSNOSTI")
    print("="*60)

    try:
        conditions = {'nm': 'Normal', 'bg': 'S torbom', 'cl': 'S kaputom'}
        datasets = {}
        
        print(f"Učitavanje podataka za analizu...")
        for code, name in conditions.items():
            X_cond, y_cond = load_condition_data(DATASET_PATH, code, target_angle='090')
            datasets[code] = (X_cond, y_cond)
            print(f"{name:<10} ({code}): {len(X_cond)} uzoraka")

        X_nm, y_nm = datasets['nm']
        if len(X_nm) == 0:
            raise Exception("Nedostaju 'nm' podaci za treniranje!")

        print("\n Treniranje modela na 'nm' podacima...")
        X_train_nm, X_test_nm, y_train_nm, y_test_nm = train_test_split(
            X_nm, y_nm, test_size=0.3, random_state=42, stratify=y_nm
        )
        
 
        model_nm = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(len(X_train_nm), 50)),
            SVC(kernel='rbf', C=10, probability=True, random_state=42)
        )
        model_nm.fit(X_train_nm, y_train_nm)
        
  
        acc_nm = accuracy_score(y_test_nm, model_nm.predict(X_test_nm))
        print(f"Baseline točnost (nm->nm): {acc_nm*100:.2f}%")

    
        final_results = {'Normal (nm)': acc_nm * 100}
        
        for code in ['bg', 'cl']:
            X_test_cond, y_test_cond = datasets[code]
            name = conditions[code]
            
            acc = 0
            if len(X_test_cond) > 0:
                
                common_subjects = np.intersect1d(y_train_nm, y_test_cond)
                mask = np.isin(y_test_cond, common_subjects)
                
                if np.sum(mask) > 0:
                    y_pred_cond = model_nm.predict(X_test_cond[mask])
                    acc = accuracy_score(y_test_cond[mask], y_pred_cond)
                    drop = (acc_nm - acc) * 100
                    print(f"Test {name} ({code}): {acc*100:.2f}% (Pad: {drop:.1f}%)")
                else:
                    print(f"{name}: Nema zajedničkih subjekata s trening skupom.")
            else:
                print(f"{name}: Nema podataka.")
            
            final_results[f'{name}\n({code})'] = acc * 100

        plt.figure(figsize=(10, 6))
        colors = ['green', 'orange', 'red']
        bars = plt.bar(final_results.keys(), final_results.values(), color=colors, alpha=0.7, edgecolor='black')
        
        plt.ylabel('Točnost (%)', fontsize=12)
        plt.title('Robusnost na promjenu uvjeta', fontsize=14)
        plt.ylim([0, 100])
        plt.axhline(y=80, color='blue', linestyle='--', label='Prag (80%)')
        
        for bar in bars:
            if bar.get_height() > 0:
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{bar.get_height():.1f}%', ha='center', fontweight='bold')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'robustness_test.png'), dpi=300)
        print(f"\nGraf spremljen: results/robustness_test.png")
        plt.show()

    except Exception as e:
        print(f"\nGreška u Fazi 5: {e}")


if __name__ == "__main__":
    main()
