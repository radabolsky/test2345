import streamlit as st
import func
from PIL import Image


st.title("Анализ эпителия")

image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
if image_file is not None:
    epit_img = Image.open(image_file)
    st.image(image_file)

    color_check = st.checkbox("Выделение площади краного цвета")
    contours_check = st.checkbox("Поиск контуров")
    artefacts_check = st.checkbox("Артефакты")

    if color_check:
        st.title("Площадь")
        k = func.get_area(epit_img)
        get_resolution = st.text_input("Разрешение (пикс/см)")
        get_scale = st.text_input("Масштаб (в мкм)")
        if get_resolution and get_scale:
            area = st.text(f'Площадь поверхности красного цвета: \n'
                           f'{k} пикс\n'
                           f'{k / (int(get_resolution) ** 2)} см^2\n'
                           f'{k * (int(get_scale) ** 2) / (int(get_resolution) ** 2)} мкм^2\n')
    if contours_check:
        st.title("Контуры")
        st.text("Поиск контуров...")
        st.image(func.get_contours(epit_img=epit_img))
    if artefacts_check:
        st.title("Артефакты")
        st.text("Поиск артефактов...")
        st.image(func.mean_shift(epit_img=epit_img))




