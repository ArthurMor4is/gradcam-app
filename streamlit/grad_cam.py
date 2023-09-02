import streamlit as st
import pandas as pd
import requests
import json
from streamlit_image_select import image_select
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

st.title("Grad-CAM Heatmap")

st.markdown(
    """
Grad-CAM is an Explainable AI technique that can be used in 
any convolutional neural network regardless of its architecture. 

I wrote a post explaining how this technique works, you can find it on [my medium page](https://medium.com/@arthurfmorais).

The implementation of this application was inspired by a [keras tutorial](https://keras.io/examples/vision/grad_cam/).

To exemplify the use of this application, select which type of object you want Grad-CAM to be applied to. 

For example, to find out if for the dog classification the neural network is highlighting the features in the image for that class select "dog" or a similar term in the field below.

Relevant features will be highlighted in red and irrelevant features in blue.
"""
)

col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.write("")

with col2:
    st.image(f"{dir_path}/images/app_diagram.png")

with col3:
    st.write("")


@st.cache
def pre_process_image(input_image):
    from tensorflow import keras

    img_size = (299, 299)
    img = keras.preprocessing.image.load_img(input_image, target_size=img_size)
    return img


# @st.cache
def get_heatmap(img, img_class):
    from tensorflow import keras

    array = keras.preprocessing.image.img_to_array(img)
    response = requests.post(
        "http://gradcam:8000/heatmap",
        json={"img_array": array.tolist(), "img_class": img_class},
    )
    return json.loads(response.text)["data"]


@st.cache
def superimpose_image(uploaded_file, heatmap, alpha=0.4):
    import numpy as np
    from matplotlib import cm
    from tensorflow import keras

    img = keras.preprocessing.image.load_img(uploaded_file)
    img = keras.preprocessing.image.img_to_array(img)

    heatmap_array = np.asarray(heatmap)
    heatmap = np.uint8(255 * heatmap_array)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img


def main(uploaded_file, selected_class):
    if uploaded_file:
        st.image(uploaded_file, caption="Original image")
        image_p = pre_process_image(uploaded_file)
        heatmap_class = get_heatmap(image_p, img_class=selected_class)
        superimposed_image = superimpose_image(uploaded_file, heatmap=heatmap_class)
        st.image(superimposed_image, caption=selected_class)


image_class = st.selectbox(
    "I want to check if the trained neural network can find a ...",
    (
        "Egyptian cat",
        "Maltese dog, Maltese terrier, Maltese",
        "Border collie",
        "Siberian husky",
        "Siamese cat, Siamese",
        "lion, king of beasts, Panthera leo",
        "bee",
        "traffic light, traffic signal, stoplight",
    ),
)

img = image_select(
    label="in this image here ...",
    images=[
        f"{dir_path}/images/cat_and_dog_1.jpeg",
        f"{dir_path}/images/lion_savana.jpeg",
        f"{dir_path}/images/traffic_light.png",
        f"{dir_path}/images/bee.jpeg",
    ],
)

main(img, image_class)

uploaded_file = st.file_uploader("Upload a photo")

main(uploaded_file, image_class)
