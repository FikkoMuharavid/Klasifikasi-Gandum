import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

st.write("""
# Aplikasi Prediksi Kelas Biji Gandum

""")

st.sidebar.header('Parameter Inputan')

def input_user():
        area = st.sidebar.slider('Area', 10.00,20.00,)
        perimeter = st.sidebar.slider('Perimeter', 10.00,17.00,)
        compactness = st.sidebar.slider('Compactness', 0.8000,0.9000,)
        length_of_Kernel = st.sidebar.slider('Length of Kernel', 5.000,6.500,)
        width_of_Kernel = st.sidebar.slider('Width of Kernel', 2.500,4.000,)
        asymmetry_Coefficient = st.sidebar.slider('Asymmetry Coefficient', 1.000,8.000,)
        length_of_Kernel_Groove = st.sidebar.slider('Length of Kernel Groove', 4.500,6.400,)
        data = {'Area' : area,
                'Perimeter' : perimeter,
                'Compactness' : compactness,
                'Length of Kernel' : length_of_Kernel,
                'Width of Kernel' : width_of_Kernel,
                'Asymmetry Coefficient' : asymmetry_Coefficient,
                'Length of Kernel Groove' : length_of_Kernel_Groove}

        fitur = pd.DataFrame(data, index=[0])
        return fitur
df = input_user()

st.subheader('Hasil Inputan')
st.write(df)

dataset = pd.read_csv('seeds.csv')
X=dataset.drop('Class',axis=1)
Y=dataset['Class']

model = LogisticRegression(C = 100)
model.fit(X, Y)

prediksi = model.predict(df)
prediksi_proba = model.predict_proba(df)

st.subheader('Keterangan Label Kelas')
jenis_seed = np.array(['0 = Kama', '1 = Rosa', '2 = Canadian'])
st.write(jenis_seed)

st.subheader('Hasil Prediksi')
if prediksi == 0:
        img = Image.open('1.png')
        img = img.resize((300, 170))
        st.image(img)
elif prediksi == 1:
        img = Image.open('2.png')
        img = img.resize((300, 170))
        st.image(img)
else:
        img = Image.open('3.png')
        img = img.resize((300, 170))
        st.image(img)

st.subheader('Probabilitas Hasil Prediksi')
st.write(prediksi_proba)