from utils import *
from download_dataset import *

"""

cd /home/rednax/SSD2TB/Github_repos/flux-trainer
conda activate flux
python caption_directory.py

"""

dataset_path = "/home/rednax/Documents/datasets/good_styles/stitchly_final/pictures_grayscale"
caption_mode = "<CAPTION>"

florence_caption_dataset(dataset_path, caption_mode)
