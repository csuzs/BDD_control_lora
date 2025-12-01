import os
import json
import pandas as pd
from tqdm import tqdm
import yaml
from pathlib import Path
import argparse

def bdd_to_hf_csv(image_folder: Path, caption_folder: Path, output_folder: Path,split: str):
    """
    Processes the dataset by matching images, condition files, and JSON captions that share the same base filename.
    Creates a CSV file that maps image paths, condition paths, and captions.
    Parameters:
    - image_folder: Path to the folder containing the image files.
    - caption_folder: Path to the folder containing the JSON files with captions.
    - output_folder: Path to the folder where the output CSV file will be saved
    - output_csv_filename: Name of the output CSV file 
    """
    assert image_folder.exists()
    assert caption_folder.exists()
    assert output_folder.exists()
    
    os.makedirs(output_folder, exist_ok=True)

    base_filenames = set(p.stem for p in image_folder.glob("*"))
    images_paths = []
    captions = []

    for file_stem in tqdm(base_filenames):
        image_path: Path = image_folder / (file_stem + ".jpg")
        caption_path: Path = caption_folder / (file_stem + ".txt") 
        
        images_paths.append(image_path)
        
        if caption_path.exists():
            with open(caption_path, 'r') as caption_file:
                caption = caption_file.read()
                #if "Close people" in caption:
                captions.append(caption)  
        else:
            captions.append("High resolution, 4k Traffic scene.")
    
    df = pd.DataFrame({
        'image': images_paths,
        'caption': captions
    })
    
    train_df = df
    
    output_csv_path: Path = output_folder / f"bdd_hf_dataset_{split}.csv"
    
    train_df.to_csv(output_csv_path, index=False)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ip','--images_path',type=str, required=True)
    parser.add_argument('-cap','--captions_path', type=str, required=True)
    parser.add_argument('-op','--output_folder', type=str, required=True)
    parser.add_argument('--split', type=str, required=True,description="Split of the dataset, e.g. train, val, test")
    
    args = parser.parse_args()
    
    bdd_to_hf_csv(image_folder=Path(args.images_path),caption_folder=Path(args.captions_path),output_folder=Path(args.output_folder),split=args.split)