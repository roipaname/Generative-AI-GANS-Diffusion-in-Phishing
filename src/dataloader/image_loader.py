import os
import tensorflow as tf
import cv2
import imghdr

data_dir="data/phishingimage"
image_ext=['jpeg','png','jpg','bmg']
print(os.listdir(data_dir))
for img_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,img_class)):
        image_path=os.path.join(data_dir,img_class)
        try:
            img=cv2.imread(image_path)
            ext=imghdr.what(image_path)
            if ext not in image_ext:
                print("removing image :",image_path)
                os.remove(image_path)
        except Exception as e:
            print("issue with :",image_path)
            os.remove(image_path)
