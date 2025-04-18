import os
import json
import pandas as pd
from tqdm import tqdm
import yaml
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

def bdd_to_hf_csv(image_folder: Path, condition_folder: Path, caption_folder: Path, output_folder: Path):
    """
    Processes the dataset by matching images, condition files, and JSON captions that share the same base filename.
    Creates a CSV file that maps image paths, condition paths, and captions.
    Parameters:
    - image_folder: Path to the folder containing the image files.
    - caption_folder: Path to the folder containing the JSON files with captions.
    - condition_folder: Path to the folder containing the condition files (segmentation masks).
    - output_folder: Path to the folder where the output CSV file will be saved
    - output_csv_filename: Name of the output CSV file 
    """
    assert image_folder.exists()
    assert caption_folder.exists()
    assert condition_folder.exists()
    assert output_folder.exists()
    
    os.makedirs(output_folder, exist_ok=True)

    base_filenames = set(p.stem for p in image_folder.glob("*"))
    images_paths = []
    conditions_paths = []
    captions = []

    for file_stem in tqdm(base_filenames):
        image_path: Path = image_folder / (file_stem + ".jpg")
        condition_path: Path = condition_folder / (file_stem + ".png") 
        caption_path: Path = caption_folder / (file_stem + ".txt") 
        
        if condition_path.exists():
            images_paths.append(image_path)
            conditions_paths.append(condition_path)
            if caption_path.exists():
                with open(caption_path, 'r') as caption_file:
                    caption = caption_file.read()
                    captions.append(caption)        
            else:
                captions.append("High resolution, 4k Traffic scene.")
        else:
            print(f"Missing files for {file_stem}, skipping.")
    
    df = pd.DataFrame({
        'image': images_paths,
        'condition': conditions_paths,
        'caption': captions
    })
    
    train_df, val_df = train_test_split(df,test_size = 0.1,random_state=42)
    train_df = df.sample(frac=0.9,random_state=42)

    val_df = df.drop(train_df.index)
    test_df = val_df.sample(frac=0.2,random_state=84)
    val_df = val_df.drop(test_df.index)
    
    output_csv_path_train: Path = output_folder / "bdd_hf_dataset_train.csv"
    output_csv_path_val: Path = output_folder / "bdd_hf_dataset_val.csv"
    output_csv_path_test: Path = output_folder / "bdd_hf_dataset_test.csv"
    
    
    train_df.to_csv(output_csv_path_train, index=False)
    val_df.to_csv(output_csv_path_val, index=False)
    test_df.to_csv(output_csv_path_test, index=False)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ip','--images_path',type=str, required=True)
    parser.add_argument('-cap','--captions_path', type=str, required=True)
    parser.add_argument('-cop','--conditions_path', type=str, required=True)
    parser.add_argument('-op','--output_folder', type=str, required=True)
    
    args = parser.parse_args()
    
    bdd_to_hf_csv(image_folder=Path(args.images_path),condition_folder=Path(args.conditions_path),caption_folder=Path(args.captions_path),output_folder=Path(args.output_folder))