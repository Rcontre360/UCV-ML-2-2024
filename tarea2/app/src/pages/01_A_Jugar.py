
import cv2
import pandas as pd
import streamlit as st
from keras.datasets import mnist
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from joblib import load
import numpy as np
import random
import io

im = Image.open("favicon.ico")
symbol_model = load('./src/models/output/symbols_model.joblib')
numbers_model = load('./src/models/output/numbers_model.joblib')

if "number" not in st.session_state:
    st.session_state["number"] = 0


@st.cache_data
def get_mnist_data():
    return mnist.load_data()


def transform_image_to_mnist(image):
    # Check if the image has 4 channels (RGBA)
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # Convertir imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
    equalized_image = cv2.equalizeHist(resized_image)

    # Retornamos la imagen transformada de INPUTxINPUT a 28x28 y la imagen con contraste
    return resized_image, equalized_image

def vectorize(original):
    return original.reshape(1,28 * 28)

def predict_number(image):
   image_reshaped = vectorize(image)
   prediction = numbers_model.predict(image_reshaped)
   return str(prediction[0])

def predict_symbol(image):
   image_reshaped = vectorize(image)
   prediction = symbol_model.predict(image_reshaped)[0]
   labels = {0: '+', 1: '-', 2: '/', 3: '/', 4: '*', 5:'*'}
   return labels[prediction]

def play_canvas():
    # Creando variables del sidebar
    number_stroke_width = 15
    exponent_stroke_width = 5
    operator_stroke_width = 8
    stroke_color = "#EEEEEE"
    bg_color = "#000000"
    realtime_update = True

    with st.container():
        (
            number_one,
            _,
            operator_one,
            number_two,
            _,
            operator_two,
            number_three,
        ) = st.columns([3, 1, 2, 3, 1, 2, 3])

        with number_one:
            c1, c2 = st.columns(2)
            with c1:
                st.empty()
            with c2:
                exponent_1 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=exponent_stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=50,
                    width=50,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="exponent_1",
                )

            number_1 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=number_stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=150,
                width=150,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="number_1",
            )

        with operator_one:
            with st.container():
                st.markdown("#")
                st.markdown("#")
                operator_1 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=operator_stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=100,
                    width=100,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="operator_1",
                )
        with number_two:
            c1, c2 = st.columns(2)
            with c1:
                st.empty()
            with c2:
                exponent_2 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=exponent_stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=50,
                    width=50,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="exponent_2",
                )
            number_2 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=number_stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=150,
                width=150,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="number_2",
            )

        with operator_two:
            st.markdown("#")
            st.markdown("#")
            operator_2 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=operator_stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=100,
                width=100,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="operator_2",
            )

        with number_three:
            c1, c2 = st.columns(2)
            with c1:
                st.empty()
            with c2:
                exponent_3 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=exponent_stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=50,
                    width=50,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="exponent_3",
                )

            number_3 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=number_stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=150,
                width=150,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="number_3",
            )

    [operacion] = st.tabs(
        ["Operacion"]
    )

    with operacion:
        st.header("Sección de Número")
        # Do something interesting with the image data and paths
        if number_1.image_data is not None:
            st.write("Operacion: ")

            number_1_img, _ = transform_image_to_mnist(number_1.image_data)
            exponent_1_img, _ = transform_image_to_mnist(exponent_1.image_data)
            operator_1_img, _ = transform_image_to_mnist(operator_1.image_data)
            number_2_img, _ = transform_image_to_mnist(number_2.image_data)
            exponent_2_img, _ = transform_image_to_mnist(exponent_2.image_data)
            operator_2_img, _ = transform_image_to_mnist(operator_2.image_data)
            number_3_img, _ = transform_image_to_mnist(number_3.image_data)
            exponent_3_img, _ = transform_image_to_mnist(exponent_3.image_data)

            number_1_pred= predict_number(number_1_img)
            exponent_1_pred= predict_number(exponent_1_img)
            operator_1_pred= predict_symbol(operator_1_img)
            number_2_pred= predict_number(number_2_img)
            exponent_2_pred= predict_number(exponent_2_img)
            operator_2_pred= predict_symbol(operator_2_img)
            number_3_pred= predict_number(number_3_img)
            exponent_3_pred= predict_number(exponent_3_img)

            full_operation = f"{number_1_pred}^{exponent_1_pred}{operator_1_pred}{number_2_pred}^{exponent_2_pred}{operator_2_pred}{number_3_pred}^{exponent_3_pred}"

            st.latex(full_operation)
            try:
                st.latex("Result: " + str(round(eval(full_operation.replace("^","**")),3)))
            except:
                st.write("Error when evaluating operation")

def main():
    play_canvas()

if __name__ == "__main__":
    main()

