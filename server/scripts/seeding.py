# This is the seeding data script
from datasets import load_dataset

ds = load_dataset("hihihohohehe/vifactcheck-normalized", split="train")

save_path = "data/vifactcheck-normalized"
ds.save_to_disk(save_path)