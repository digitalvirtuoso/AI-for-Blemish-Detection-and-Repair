import os
import cv2
from pathlib import Path

path = Path('../data/')
raw_path = path / 'raw'
prep_path = path / 'preprocessed'

# Preprocessing images
def process_raw(raw_path, prep_path):
    for filename in os.listdir(raw_path):
            raw = str(raw_path/filename)
            prep = str(prep_path/filename)
            image = cv2.imread(raw)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(prep, gray_image)
            
process_raw(raw_path, prep_path)