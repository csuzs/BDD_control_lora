import numpy as np
import os
import argparse
from tqdm import tqdm

def select_with_numpy(lst):
    n = len(lst)
    sample_size = max(1, n // 10)
    available_indices = np.arange(n)
    results = []
    
    for i in range(10):
        if len(available_indices) < sample_size:
            selected_indices = available_indices.copy()
            available_indices = np.array([])
        else:
            selected_indices = np.random.choice(
                available_indices, 
                size=sample_size, 
                replace=False
            )
            available_indices = np.setdiff1d(available_indices, selected_indices)
        
        selected_items = [lst[idx] for idx in selected_indices]
        results.append(selected_items)
        
        if len(available_indices) == 0:
            break
    
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=r'Split BDD100K dataset into 10% groups for accuracy analysis.')
    parser.add_argument('--bdd_data_path', type=str, required=True, help='Path to BDD100K root directory (should contain images/100k/train and labels/det_yoloformat/labels/train)')
    parser.add_argument('--output_dir', type=str, default='bdd_10p_split_dset', help='Output directory for split groups')
    args = parser.parse_args()

    images_dir = os.path.join(args.bdd_data_path, "images", "100k", "train")
    label_dir = os.path.join(args.bdd_data_path, "labels", "det_yoloformat", "labels", "train")
    split_output_base_dir = args.output_dir

    lst = list(
        map(lambda entry: entry.path,
            os.scandir(images_dir)
        )
    )

    results = select_with_numpy(lst)

    img_l_pairs = []
    for group in results:
        group_pairs = []
        for img_path in group:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, img_name + ".txt")
            group_pairs.append((img_path, label_path))
        img_l_pairs.append(group_pairs)

    for group_idx, group in tqdm(enumerate(img_l_pairs)):
        grp_dir = os.path.join(split_output_base_dir, f"grp_{group_idx}")

        os.makedirs(grp_dir, exist_ok=True)
        img_train_path = os.path.join(grp_dir, "images", "train")
        label_train_path = os.path.join(grp_dir, "labels", "train")
        os.makedirs(img_train_path, exist_ok=True)
        os.makedirs(label_train_path, exist_ok=True)

        for img_path, label_path in group:
            img_name = os.path.basename(img_path)
            new_img_path = os.path.join(img_train_path, img_name)
            new_label_path = os.path.join(label_train_path, os.path.splitext(img_name)[0] + ".txt")

            os.system(f"cp {img_path} {new_img_path}")
            if os.path.exists(label_path):
                os.system(f"cp {label_path} {new_label_path}")
