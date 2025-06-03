import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Helper function to compute stats for one image
def compute_image_stats_row(row_dict, bf_path, fl_path):
    try:
        name = row_dict['Name']
        label = row_dict['Diagnosis']
        patient_id = row_dict['patient_id']

        bf = np.array(Image.open(os.path.join(bf_path, name)).convert('L'))
        fl = np.array(Image.open(os.path.join(fl_path, name)).convert('L'))

        return [
            {
                'image': name,
                'patient_id': patient_id,
                'type': 'BF',
                'label': label,
                'mean': bf.mean(),
                'min': bf.min(),
                'max': bf.max(),
                'std': bf.std()
            },
            {
                'image': name,
                'patient_id': patient_id,
                'type': 'FL',
                'label': label,
                'mean': fl.mean(),
                'min': fl.min(),
                'max': fl.max(),
                'std': fl.std()
            }
        ]
    except Exception as e:
        print(f"[ERROR] Failed on {name}: {e}")
        return []

# Parallelized main function
def compute_image_intensity_stats(df, bf_path, fl_path, output_csv='image_intensity_stats.csv', max_workers=6):
    stats = []
    print(f"Starting processing with ProcessPoolExecutor")

    row_dicts = df.to_dict(orient='records')  # Convert rows to list of dicts for pickling
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_image_stats_row, row, bf_path, fl_path): row for row in row_dicts}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            result = future.result()
            if result:
                stats.extend(result)

    print(f"Done processing with ProcessPoolExecutor")
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_csv, index=False)
    print(f"Saved intensity statistics to {output_csv}")


import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Compute stats for one test image pair
def compute_image_stats_test(name, bf_path, fl_path):
    try:
        bf = np.array(Image.open(os.path.join(bf_path, name)).convert('L'))
        fl = np.array(Image.open(os.path.join(fl_path, name)).convert('L'))

        return [
            {
                'image': name,
                'type': 'BF',
                'mean': bf.mean(),
                'min': bf.min(),
                'max': bf.max(),
                'std': bf.std()
            },
            {
                'image': name,
                'type': 'FL',
                'mean': fl.mean(),
                'min': fl.min(),
                'max': fl.max(),
                'std': fl.std()
            }
        ]
    except Exception as e:
        print(f"[ERROR] Failed on {name}: {e}")
        return []

# Main function to process all test images
def compute_test_image_intensity_stats_from_folder(
    fl_path,
    bf_path,
    output_csv='test_image_intensity_stats.csv',
    max_workers=6
):
    print("Scanning test images...")

    image_names = sorted([f for f in os.listdir(fl_path) if f.lower().endswith(('.jpg'))])

    print(f"Found {len(image_names)} FL test images. Starting processing...")

    stats = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compute_image_stats_test, name, bf_path, fl_path): name
            for name in image_names
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing test images"):
            result = future.result()
            if result:
                stats.extend(result)

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_csv, index=False)
    print(f"Saved test image statistics to: {output_csv}")
