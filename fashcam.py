import streamlit as st
from PIL import Image
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import webbrowser
import pandas as pd
from keras.models import Model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Sequential, model_from_json
import os
from io import StringIO,BytesIO
from keras.applications.imagenet_utils import preprocess_input
import cv2

#styles=pd.read_csv(r'E:/Data Science Bootcamp/Final Project/fashion_dataset/styles.csv')
#output_img_file = r'E:/Data Science Bootcamp/Final Project/fashion_dataset/images'
#image_names=os.listdir(output_img_file)

unique_types = ['Jeans', 'Shirts', 'Watches', 'Track Pants', 'Tshirts', 'Socks', 'Casual Shoes', 'Belts', 'Flip Flops', 'Handbags', 'Tops', 'Bra', 'Sandals', 'Shoe Accessories', 'Sweatshirts', 'Deodorant', 'Formal Shoes', 'Bracelet', 'Lipstick', 'Flats', 'Kurtas', 'Waistcoat', 'Sports Shoes', 'Shorts', 'Briefs', 'Sarees', 'Perfume and Body Mist', 'Heels', 'Sunglasses', 'Innerwear Vests', 'Pendant', 'Nail Polish', 'Laptop Bag', 'Scarves', 'Rain Jacket', 'Dresses', 'Night suits', 'Skirts', 'Wallets', 'Blazers', 'Ring', 'Kurta Sets', 'Clutches', 'Shrug', 'Backpacks', 'Caps', 'Trousers', 'Earrings', 'Camisoles', 'Boxers', 'Jewellery Set', 'Dupatta', 'Capris', 'Lip Gloss', 'Bath Robe', 'Mufflers', 'Tunics', 'Jackets', 'Trunk', 'Lounge Pants', 'Face Wash and Cleanser', 'Necklace and Chains', 'Duffel Bag', 'Sports Sandals', 'Foundation and Primer', 'Sweaters', 'Free Gifts', 'Trolley Bag', 'Tracksuits', 'Swimwear', 'Shoe Laces', 'Fragrance Gift Set', 'Bangle', 'Nightdress', 'Ties', 'Baby Dolls', 'Leggings', 'Highlighter and Blush', 'Travel Accessory', 'Kurtis', 'Mobile Pouch', 'Messenger Bag', 'Lip Care', 'Face Moisturisers', 'Compact', 'Eye Cream', 'Accessory Gift Set', 'Beauty Accessory', 'Jumpsuit', 'Kajal and Eyeliner', 'Water Bottle', 'Suspenders', 'Lip Liner', 'Robe', 'Salwar and Dupatta', 'Patiala', 'Stockings', 'Eyeshadow', 'Headband', 'Tights', 'Nail Essentials', 'Churidar', 'Lounge Tshirts', 'Face Scrub and Exfoliator', 'Lounge Shorts', 'Gloves', 'Mask and Peel', 'Wristbands', 'Tablet Sleeve', 'Ties and Cufflinks', 'Footballs', 'Stoles', 'Shapewear', 'Nehru Jackets', 'Salwar', 'Cufflinks', 'Jeggings', 'Hair Colour', 'Concealer', 'Rompers', 'Body Lotion']

st.set_page_config(page_title="Image Recommendation System",layout="wide")

col1,mid,col2 = st.columns([1,15,100])
with col1:
    st.image('./Fashion_Camera2.jpg', width=150)
with col2:
    #st.write('A Name')
    st.markdown('<h1 style="color: red;font-size: 70px;">FashCam</h1>',
                            unsafe_allow_html=True)
    st.markdown('<h1 style="color: black;font-size: 30px;">...an Image Search Engine</h1>',
                            unsafe_allow_html=True)
    
    st.markdown("Our idea is to build a new search engine **:red[_FashCam_]**:camera:.We have developed a cutting-edge image recognition technology that makes it easy to find the fashion you want. With **:red[_FashCam_]**:camera:, you can simply take a picture of an item or upload an image and our algorithm will match it with similar products available for purchase online. It's that simple!")
st.sidebar.write("## Upload or Take a Picture")


    
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model("./model_fashion.h5")
    #return model

with st.spinner('Please wait, while the model is being loaded..'):
    model=load_model()

# Upload the image file
uploaded_file = st.sidebar.file_uploader(
"Choose an image from your computer", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.sidebar.subheader("Please upload a product image using the browse button :point_up:")
    st.sidebar.write("Sample image can be found [here](https://github.com/prachiagrl83/WBS/tree/Prachi/Sample_images)!")
    
else:
    st.sidebar.subheader("Thank you for uploading the image. Below you can see image which you have just uploaded!")
    st.subheader("Scroll down to see the Top Similar Products...")  
    st.sidebar.image(uploaded_file,width=250)
    image_data = uploaded_file.getvalue()
    #bytes_data = img_file_buffer.getvalue()
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    # Check the type of cv2_img:
    # Should output: <class 'numpy.ndarray'>
    st.write(type(img))
    # Check the shape of cv2_img:
    # Should output shape: (height, width, channels)
    st.write(img.shape)

#img = load_img(uploaded_file)
#st.image(img,width=250)

def load_image(image):
    img=tf.image.decode_jpeg(image,channels=3)
    img=tf.cast(img,tf.float32)
    img/=255.0
    img=tf.image.resize(img,(28,28,1))
    img=tf.expand_dims(img,axis=0)
    return img
    #st.image(load_image(img))
    
def get_prediction_tuning():
    # load json and create model
    json_file = open('model_fashion.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_weights_fashion.h5")
    # Load an image to use for prediction
    #img = load_img(cv2_img, target_size=(224,224,3))
    #img = cv2_img.reshape(224,224,3)
    #img = load_img(uploaded_file)
    #x = img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    x = preprocess_input(load_image(img))
    result = loaded_model.predict(x)
    # evaluate loaded model on test data
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy',
                         metrics=['accuracy'])
    string = get_prediction_tuning(unique_types[np.argmax(result)[1]])
    st.success(string)
    st.image(get_prediction_tuning(unique_types[np.argmax(result)[1]]))