import numpy as np
import pandas as pd
import pickle
import os
import re

# 1. Path input
data_path = 'dataset/ntu25_data/balanced/joint/train_data.npy'
label_path = 'dataset/ntu25_data/balanced/joint/train_label.pkl'
output_csv = 'docs/ntu_balanced_dataset_summary.csv'

def get_action_details(sample_name):
    match = re.search(r'A(\d{3})', sample_name)
    if not match:
        return "Unknown", "Unknown", "Unknown"
    
    action_id = f"A{match.group(1)}"
    
    mapping = {
        "A043": ("falling_down", "fall"),
        "A008": ("sitting_down", "not_fall"),
        "A009": ("standing_up", "not_fall"),
        "A027": ("jumping_up", "not_fall"),
        "A042": ("staggering", "not_fall"),
    }
    
    action_name, category = mapping.get(action_id, ("other", "not_fall"))
    return action_id, action_name, category

def count_valid_frames(sample_tensor):
    # sample_tensor shape: (C, T, V, M)
    # T di sini berasal dari dimensi ke-2 file .npy Anda (hasil max_frames saat prep)
    conf = sample_tensor[2, :, :, 0] 
    valid = int((conf > 0).any(axis=1).sum())
    return max(valid, 1)

def export_summary():
    if not os.path.exists(data_path):
        print(f"Error: {data_path} tidak ditemukan!")
        return

    data = np.load(data_path, mmap_mode='r')
    with open(label_path, 'rb') as f:
        sample_names, labels = pickle.load(f)

    N, C, T, V, M = data.shape
    print(f"Mengekspor {N} sampel dengan durasi T={T} frame...")

    summary_rows = []
    
    # Header untuk kejelasan
    header = [
        "Sample_Name", "Split", "Action_ID", "Action_Name", 
        "Category", "Valid_Frames", "Channels(C)", "Joints(V)", 
        "Max_Frames(T)", "Persons(M)"
    ]

    for n in range(N):
        name = sample_names[n]
        display_name = name if name.endswith("_rgb") else f"{name}_rgb"
        action_id, action_name, category = get_action_details(name)
        valid_frames = count_valid_frames(data[n])
        
        row = [
            display_name, "train", action_id, action_name, 
            category, valid_frames, C, V, T, M
        ]
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows, columns=header)
    df.to_csv(output_csv, index=False) # index=False agar tidak ada kolom angka di paling kiri
    
    print(f"Selesai! Header telah ditambahkan.")
    print(f"T={T} diambil langsung dari dimensi file .npy Anda.")

if __name__ == "__main__":
    export_summary()
