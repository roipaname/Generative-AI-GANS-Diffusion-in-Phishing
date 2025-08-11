from src.dataloader.image_loader import load_and_preprocess_data

data_dir="data/phishingimage"
X,y,class_label=load_and_preprocess_data(data_dir)
print(X)