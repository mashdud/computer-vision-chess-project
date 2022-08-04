import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

mapping_dict = {
    "k": 1,
    "q": 2,
    "r": 3,
    "n": 4,
    "b": 5,
    "p": 6
}

dict_mapping = {
    1: "k",
    2: "q",
    3: "r",
    4: "n",
    5: "b",
    6: "p"
}


def encode_fen(fen):
    y = []
    fen = fen.replace("-", "")
    for char in fen:
        if char.isalpha():
            y.append(mapping_dict[char.lower()] + 6 if char.isupper() else mapping_dict[char])
        else:
            y.extend(np.zeros(int(char), np.int16).tolist())
    return y


def decode_fen(encoded_array):
    step = 8
    result = []
    for i in range(8):
        extracted_label = encoded_array[i * step:i * step + step]
        zero_count = 0
        concatenated_label = ''
        for index, num in enumerate(extracted_label):
            if zero_count > 0 and num != 0:
                concatenated_label += str(zero_count)
                zero_count = 0
            if num == 0:
                zero_count += 1
            elif 1 <= num <= 6:
                concatenated_label += dict_mapping[num]
            elif num == 12:
                concatenated_label += dict_mapping[6].upper()
            else:
                concatenated_label += dict_mapping[num % 6].upper()
            if index == len(extracted_label) - 1:
                concatenated_label += str(zero_count)
                zero_count = 0
        result.append(concatenated_label)
    return "-".join(result)


IMG_SIZE = 200


def process_image(img):
    new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    new_array = cv2.equalizeHist(new_array)
    new_array = cv2.equalizeHist(new_array)
    return new_array.reshape(IMG_SIZE, IMG_SIZE, 1)


def pre_process(file):
    x = []
    y = []
    x.append(process_image(cv2.imread(file, cv2.IMREAD_GRAYSCALE)))
    y.append(encode_fen(os.path.basename(file).split(".")[0]))
    return x, y


uploaded_file = st.file_uploader("Upload File", type=['png', 'jpeg', 'jpg'])
if uploaded_file:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    model = tf.keras.models.load_model('chess_prediction.h5')
    data, label = pre_process(uploaded_file.name)
    st.write(decode_fen(model.predict(np.array(data)).astype(int)[0]))

fen_label = "1B1R4-3b1n2-5K2-3p4-3b3R-Q4p2-4n3-2b1bRk1"
print(encode_fen(fen_label))
print(decode_fen(encode_fen(fen_label)))
# print(fen_label)