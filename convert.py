import json
import random
import string

def convert_to_karpathy_json(txt_path, output_json_path):
    images_dict = {}
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
            
        first_comma_idx = line.find(',')
        if first_comma_idx == -1:
            continue
            
        img_name = line[:first_comma_idx]
        caption = line[first_comma_idx + 1:]
        
        if img_name not in images_dict:
            images_dict[img_name] = []
        images_dict[img_name].append(caption)

    karpathy_format = {"images": []}
    img_id = 0
    
    for img_name, captions in images_dict.items():
        rand_num = random.random()
        if rand_num < 0.8:
            split = 'train'
        elif rand_num < 0.9:
            split = 'val'
        else:
            split = 'test'
            
        sentences = []
        for cap in captions:
            tokens = cap.lower().translate(str.maketrans('', '', string.punctuation)).strip().split()
            sentences.append({
                "tokens": tokens,
                "raw": cap
            })
            
        image_data = {
            "filepath": "flickr8k", 
            "filename": img_name,
            "imgid": img_id,
            "split": split,
            "sentences": sentences
        }
        karpathy_format["images"].append(image_data)
        img_id += 1
        
    with open(output_json_path, 'w') as f:
        json.dump(karpathy_format, f)

convert_to_karpathy_json('captions.txt', 'dataset_flickr8k.json')