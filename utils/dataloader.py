import os
import shutil
import pandas as pd

df = pd.read_csv("/srv/scratch1/swallace/CancerSeg/data/train.csv")
label_dirs = "/srv/scratch1/swallace/CancerSeg/data/cancerous"

if not os.path.exists(label_dirs):
    os.makedirs(label_dirs)
    
    
for _, row in df.iterrows():
    image_name = row['Name']
    print(image_name)
    label = "Cancerous" if row['Diagnosis'] else "Clean"

    label_dir = os.path.join(label_dirs, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    src_path = os.path.join('/srv/scratch1/swallace/CancerSeg/data/FL/train', image_name)
    dest_path = os.path.join(label_dir, image_name)
    
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
        print(f"Moved {image_name}, to {label_dir}")
    else:
        print(f"WARNING: {src_path} does not exist")

print("Process completed!")