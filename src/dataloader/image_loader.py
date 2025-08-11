import os
import tensorflow as tf

data_dir="data/phishingimage"
image_ext=['jpeg','png','jpg','bmg']
print(os.listdir(data_dir))
for img_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,img_class)):
        print(image)