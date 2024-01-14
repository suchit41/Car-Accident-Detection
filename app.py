import streamlit as st
import os 
import dill
from pathlib import Path
from fastai.vision.all import load_learner, PILImage
from PIL import Image
import numpy as np

learn = load_learner('crashornot.pkl', cpu=False, pickle_module=dill)

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])
if uploaded_file is not None:
   image = Image.open(uploaded_file)
   st.image(image, caption='Uploaded Image.', use_column_width=True)
   if st.button('Predict'):
       image = PILImage.create(uploaded_file)
       is_crash, _, probs = learn.predict(image)
       st.write(f"This is a: {is_crash}.")
       st.write(f"Probability it's a crash: {probs[0]:.4f}")
