from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report
from tensorflow import keras
import numpy as np

#image picker import
from tkinter import Tk
from tkinter.filedialog import askopenfilename


# pick the image
Tk().withdraw()
file_dir = askopenfilename()



# load the trained model through keras
model = load_model('model')
image_result=Image.open(file_dir)
from tensorflow.keras.preprocessing import image
file_dir=image.load_img(file_dir,target_size=(224,224))
file_dir=image.img_to_array(file_dir)
file_dir=file_dir/255
file_dir=np.expand_dims(file_dir,axis=0 )
result=model.predict(file_dir)
print (result)
print(np.argmax(result))
categories=['Blight','Common_Rust','Gray_Leaf_Spot','Healthy']
image_result=plt.imshow(image_result)
plt.title(categories[np.argmax(result)])
plt.show()