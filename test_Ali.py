import streamlit as st
import tensorflow.keras as keras
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import cv2

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

def load():
    model=load_model("./model_fashion.h5")
    return model

with st.spinner('Please wait, while the model is being loaded..'):
    model=load()

file = st.sidebar.file_uploader("Please upload a product image.", type=["jpg", "png"])


def import_and_predict(image_data,model):
    size = (28, 28)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #img_reshape = img[np.newaxis, ...]
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    prediction = model.predict(x)
    return prediction


if file is None:
    st.sidebar.text("No image file has been uploaded.")
else:
    image = Image.open(file)
    st.sidebar.image(image,width=250)
    predictions = import_and_predict(image,model)
    #class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    #result = model.predict(x)
    # evaluate loaded model on test data
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                         metrics=['accuracy'])
    #return (result,unique_types[np.argmax(result)])
    string = "The Similar products are: " + unique_types[np.argmax(predictions)[1]]
    st.success(string)
    st.image(image)