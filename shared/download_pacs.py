import os
from datasets import load_dataset

OUT_ROOT = r"C:\Users\RBTG\Desktop\ATML_PA1\PACS_data\kfold"

print("Downloading PACS from Hugging Face (flwrlabs/pacs)...")
ds = load_dataset("flwrlabs/pacs", split="train")

print(f"Loaded {len(ds)} images. Writing to disk in kfold/<domain>/<class>/ layout...")
label_names = ds.features["label"].names  # e.g. ['dog','elephant',...]

for i, example in enumerate(ds):
    domain = example["domain"]
    class_name = label_names[example["label"]]
    out_dir = os.path.join(OUT_ROOT, domain, class_name)
    os.makedirs(out_dir, exist_ok=True)
    example["image"].save(os.path.join(out_dir, f"img_{i:05d}.jpg"))
    if i % 1000 == 0:
        print(f"  {i}/{len(ds)} written...")

print("Done. PACS written to:", OUT_ROOT)