from PIL import Image
from lang_sam import LangSAM

model = LangSAM()
image_pil = Image.open("/data/xander/Projects/cog/GitHub_repos/flux-trainer/datasets/banny_best/Juiceisflowing.jpg").convert("RGB")
text_prompt = "yellow cartoon banana."
results = model.predict([image_pil], [text_prompt])