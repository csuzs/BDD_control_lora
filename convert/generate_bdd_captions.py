import json
import os


def create_bdd_captions(frame_metadata_path: str, caption_txt_output_folder: str):
    with open(frame_metadata_path) as f:
        mtdt_json = json.load(f)
        for i in range(len(mtdt_json)):
            prompt = "Traffic scene."
            
            frame_mtdt = mtdt_json[i]
            
            if frame_mtdt:
                frame_name = frame_mtdt['name'].split(".")[0] 
                attributes = frame_mtdt['attributes']
                if attributes["weather"] != "undefined":
                    prompt += f"{attributes["weather"]} weather. "
                
                if attributes["timeofday"] != "undefined":
                    prompt += f"{attributes["timeofday"]}. "
                
                if attributes["scene"] != "undefined":
                    prompt += f"{attributes["scene"]}."
            else:
                prompt.rstrip()
                
            with open(f"{caption_txt_output_folder}{os.sep}{frame_name}.txt","w") as cap_file:
                cap_file.write(prompt)

if __name__ == "__main__":
    create_bdd_captions(frame_metadata_path="bdd/bdd100k/labels/det_20/det_train.json",caption_txt_output_folder="bdd_captions")