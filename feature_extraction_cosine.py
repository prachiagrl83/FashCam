# TensorFlow and tf.keras
import streamlit as st
from keras.applications import ResNet50
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import Model
from PIL import Image
import numpy as np
from keras.models import Model
import os
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

import os
from tensorflow.keras.applications.resnet50 import ResNet50
unique_types = ['Backpacks',
                'Belts',
                'Bra',
                'Caps-hats',
                'Casual Shoes',
                'Dresses',
                'Earrings',
                'Handbags',
                'Heels',
                'Leggings',
                'Outwears',
                'pijamas',
                'Ring',
                'Sandals',
                'Scarves',
                'Shirts',
                'Shorts',
                'Skirts',
                'Sportswear',
                'Sunglasses',
                'Sweatshirts',
                'Tops',
                'Trousers',
                'Tshirts']
# model = load_model('final_model_v2.h5')
# function that takes an image and available_class as input and applies the cosine similarity with the features from csv to get the closest images


def get_closest_images(img_reshape, available_class):
    nb_closest_images = 5

    features = np.genfromtxt(f'{available_class}.csv', delimiter=',')
    # get the features of the image
    model = ResNet50(weights='imagenet', include_top=True)
    feat_extractor = Model(
        inputs=model.input, outputs=model.get_layer("avg_pool").output)
    feat_extractor.summary()
    img_features = feat_extractor.predict(img_reshape)
    # get the cosine similarity
    cosSimilarities = cosine_similarity(img_features, features)
    closest_imgs_indexes = cosSimilarities.argsort()[0][-nb_closest_images:]
    # similarity score of the closest images
    closest_imgs_similarities = cosSimilarities[0][closest_imgs_indexes]
    shopfiles = [f'finalDataset/{available_class}/' +
                 f for f in os.listdir(f'finalDataset/{available_class}')]
    # get the closest images
    closest_imgs = [shopfiles[i]
                    for i in closest_imgs_indexes if i < len(shopfiles)]
    similar = []
    for i in range(0, nb_closest_images):
        # selecting the most similar with threshold 0.3
        if closest_imgs_similarities[i] > 0.4:
            # give the path of the image
            similar.append(closest_imgs[i])

    return similar[:3]