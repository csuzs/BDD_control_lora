import json
import os
from pathlib import Path


def create_bdd_captions(frame_metadata_path: str, caption_txt_output_folder: str):
    with open(frame_metadata_path) as f:
        mtdt_json = json.load(f)
        for i in range(len(mtdt_json)):
            has_ped = False
            has_cycle = False
            has_buses = False
            
            frame_mtdt = mtdt_json[i]
            try:
                frame_labels = frame_mtdt['labels']
                has_ped: bool = any(label['category'] == 'pedestrian' for label in frame_labels)
                has_cycle: bool =any(label['category'] == 'bycicle' for label in frame_labels)
                
                has_motorcycle: bool =any(label['category'] == 'motorcycle' for label in frame_labels)
                
                has_buses: bool =any(label['category'] == "bus" for label in frame_labels)  
                has_train: bool =any(label['category'] == "train" for label in frame_labels)
                
                ped_boxes = [label["box2d"] for label in frame_labels if label["category"] == "pedestrian"]
                is_large_ped = any(
                    (box["x2"] - box["x1"] > 75 or box["y2"] - box["y1"] > 150) for box in ped_boxes
                )
                
            except:
                is_large_ped = False
            
            prompt = "High resolution, 4k Traffic scene. "
            if is_large_ped:
                prompt += "pedestrians walking close to the car. "
            elif has_ped:
                prompt += "Pedestrians walking. "
            if has_cycle:
                prompt += "Bycicles on the road. "
            if has_motorcycle:
                prompt += "Motorcycles on the road. "
            if has_train:
                prompt += "Train next to the road. "
            if has_buses:
                prompt += "Buses on the road. "
            
            frame_name = frame_mtdt['name'].split(".")[0]
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
            
            prompt.rstrip()

            with open(f"{caption_txt_output_folder}{os.sep}{frame_name}.txt","w") as cap_file:
                cap_file.write(prompt)

if __name__ == "__main__":
    bdd_prefix_path = os.environ.get("BDD_DATA_DIR","")
    create_bdd_captions(frame_metadata_path=Path(bdd_prefix_path) / Path("det_val.json"),caption_txt_output_folder="/storage/gpfs/data-store/projects/parking-data-ops/ws/shared/project-workspace/uic19759/bdd_captions/10k/bdd_captions_all_relevant_objects/val")