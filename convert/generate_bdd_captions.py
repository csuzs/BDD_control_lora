import json
import os
from pathlib import Path


def create_bdd_captions(frame_metadata_path: str, caption_txt_output_folder: str):
    with open(frame_metadata_path) as f:
        mtdt_json = json.load(f)
        for i in range(len(mtdt_json)):
            prompt = "High resolution, 4k Traffic scene. "
            
            frame_mtdt = mtdt_json[i]
            
            if frame_mtdt:
                frame_name = frame_mtdt['name'].split(".")[0] 
                attributes = frame_mtdt['attributes']
                if attributes["weather"] != "undefined":
                    prompt += f"{attributes["weather"]} weather. ".capitalize()
                
                if attributes["timeofday"] != "undefined":
                    prompt += f"{attributes["timeofday"]}. ".capitalize()
                
                if attributes["scene"] != "undefined":
                    prompt += f"{attributes["scene"]}.".capitalize()
            else:
                prompt.rstrip()
                
            with open(f"{caption_txt_output_folder}{os.sep}{frame_name}.txt","w") as cap_file:
                cap_file.write(prompt)

if __name__ == "__main__":
    bdd_prefix_path = os.environ.get("BDD_DATA_DIR","")
    create_bdd_captions(frame_metadata_path=Path(bdd_prefix_path) / Path("bdd/bdd100k/labels/det_20/det_train.json"),caption_txt_output_folder="bdd_captions")