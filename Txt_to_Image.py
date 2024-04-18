import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Txt to Image",
    page_icon="🧊",)

@st.cache_data  # makes image update faster
def txt2array(txt_file: str) -> np.ndarray:
    """Used in the store_in_hdf5 method.
    Convert a .txt file to a 2D numpy array."""
    data = np.loadtxt(txt_file, skiprows=19)

    # Step 1: Determine the dimensions of the image
    max_row = int(np.max(data[:, 0])) + 1  # Adding 1 because index starts at 0
    max_col = int(np.max(data[:, 1])) + 1  # Adding 1 because index starts at 0

    # Step 2: Create an empty 2D array
    image_array = np.zeros((max_row, max_col), dtype=np.uint16)

    # Step 3: Fill the array with pixel values
    for row in data:
        r, c, val = int(row[0]), int(row[1]), row[2]
        image_array[r, c] = val + 1

    return image_array


def plot_image(image_array: np.ndarray) -> None:
    """Plot a 2D numpy array as an image."""
    figure = px.imshow(image_array)

    return figure



st.title("Pockels image viewer")

color_theme = st.sidebar.radio(
    "Choose a color theme:", ("Viridis", "Plasma", "Inferno", "Jet"), index=0
)

reverse_color_theme = st.sidebar.checkbox("Reverse color theme")

if reverse_color_theme:
    color_theme = color_theme + "_r"

uploaded_file = st.sidebar.file_uploader("Please upload a .txt file: ", type=["txt"], key="file1")
uploaded_file_2 = st.sidebar.file_uploader("Please upload a .txt file: ", type=["txt"], key="file2")


# checks if file is uploaded
if uploaded_file is not None:
    st.subheader("Image 1:")

    # convert bytes to a numpy array
    image_array_1 = txt2array(uploaded_file).astype(int)

    # making float min/max values of the slider based on min/max values of image array
    min_val, max_val = st.slider(
        label="Color range slider:",
        min_value=0,
        max_value=int(image_array_1.max()),
        value=[0, 700],
        step=10,
    )

    figure = px.imshow(
        image_array_1, color_continuous_scale=color_theme, range_color=[min_val, max_val]
    )

    st.plotly_chart(figure)


# checks if file is uploaded
if uploaded_file_2 is not None:
    st.subheader("Image 2:")

    # convert bytes to a numpy array
    image_array_2 = txt2array(uploaded_file_2).astype(int)

    # making float min/max values of the slider based on min/max values of image array
    min_val, max_val = st.slider(
        label="Color range slider:",
        min_value=0,
        max_value=int(image_array_2.max()),
        value=[0, 700],
        step=10,
    )

    figure2 = px.imshow(
        image_array_2, color_continuous_scale=color_theme, range_color=[min_val, max_val]
    )

    st.plotly_chart(figure2)


if uploaded_file is not None and uploaded_file_2 is not None:

    st.subheader("Image 1 - Image 2:")

    difference_image = image_array_1 - image_array_2

    # convert difference image to csv
    data = pd.DataFrame(difference_image)
    csv = data.to_csv(index=False).encode('utf-8')

    # download button for the difference image
    st.download_button(
        label="Download difference image",
        data=csv,
        file_name="difference_image.csv",
        mime="text/csv",
    )

    # making float min/max values of the slider based on min/max values of image array
    min_val, max_val = st.slider(
        label="Color range slider:",
        min_value=int(difference_image.min()),
        max_value=int(difference_image.max()),
        value=[0, 700],
        step=10,
    )

    figure3 = px.imshow(
        difference_image, color_continuous_scale=color_theme, range_color=[min_val, max_val])

    st.plotly_chart(figure3)
