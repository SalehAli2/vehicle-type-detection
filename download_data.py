import urllib.request
import zipfile
import os

url = "https://universe.roboflow.com/ds/m6ObF7cxVw?key=7s9quO39Ll"
print("Downloading dataset...")
urllib.request.urlretrieve(url, "dataset.zip")

print("Extracting...")
with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
    zip_ref.extractall("data/raw")

os.remove("dataset.zip")
print("Done! Dataset saved to data/raw/")