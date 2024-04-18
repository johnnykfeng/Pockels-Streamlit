import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Transmission Calculator")

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

st.title("Transmission Image Calculator")

color_theme = st.sidebar.radio(
    "Choose a color theme:", ("Viridis", "Plasma", "Inferno", "Jet"), index=0
)

reverse_color_theme = st.sidebar.checkbox("Reverse color theme")

if reverse_color_theme:
    color_theme = color_theme + "_r"

st.sidebar.subheader("Upload the 'Dark Parallel', '0V Parallel', and '0V Crossed' for calibration:")
upload_dark = st.sidebar.file_uploader("**'Dark Parallel' image**", type=["txt"], key="dark")
upload_parallel = st.sidebar.file_uploader("**'0V Parallel' image**", type=["txt"], key="parallel")
upload_crossed = st.sidebar.file_uploader("**'0V Crossed' image**", type=["txt"], key="crossed")
upload_biased = st.sidebar.file_uploader("**'Biased' image**", type=["txt"], key="biased")

# col1, col2, col3 = st.columns([1, 1, 1])

if upload_dark is not None:

    dark_image = txt2array(upload_dark).astype(int)
    
    with st.expander("Dark image", expanded=True):

        st.subheader("Dark image:")

        min_val, max_val = st.slider(
            label="Color range slider:",
            min_value=0,
            max_value=500,
            value=[0, int(dark_image.max())],
            step=5,
            key = "dark_slider"
        )

        figure = px.imshow(
            dark_image, color_continuous_scale=color_theme, range_color=[min_val, max_val]
        )

        st.plotly_chart(figure)

if upload_parallel is not None:

    parallel_image = txt2array(upload_parallel).astype(int)

    with st.expander("0V Parallel image", expanded=True):
        st.subheader("0V Parallel image:")

        min_val, max_val = st.slider(
            label="Color range slider:",
            min_value=0,
            max_value=500,
            value=[0, int(parallel_image.max())],
            step=5,
            key="parallel_slider"
        )

        figure2 = px.imshow(
            parallel_image, color_continuous_scale=color_theme, range_color=[min_val, max_val]
        )

        st.plotly_chart(figure2)
        
if upload_crossed is not None:
    
        crossed_image = txt2array(upload_crossed).astype(int)

        with st.expander("0V Crossed image", expanded=True):
        
            st.subheader("0V Crossed image:")

            min_val, max_val = st.slider(
                label="Color range slider:",
                min_value=0,
                max_value=500,
                value=[0, int(crossed_image.max())],
                step=5,
                key="crossed_slider"
            )

            figure3 = px.imshow(
                crossed_image, color_continuous_scale=color_theme, range_color=[min_val, max_val]
            )

            st.plotly_chart(figure3)

if upload_biased is not None:
        
        biased_image = txt2array(upload_biased).astype(int)
    
        with st.expander("Biased image", expanded=True):
    
            st.subheader("Biased image:")
    
            min_val, max_val = st.slider(
                label="Color range slider:",
                min_value=0,
                max_value=500,
                value=[0, int(biased_image.max())],
                step=5,
                key="biased_slider"
            )
    
            figure4 = px.imshow(
                biased_image, color_continuous_scale=color_theme, range_color=[min_val, max_val]
            )
    
            st.plotly_chart(figure4)

if upload_dark is not None and upload_parallel is not None and upload_crossed is not None and upload_biased is not None:
    
    st.subheader("Transmission image:")
    # convert all images to float
    # dark_image = dark_image.astype(float)
    # parallel_image = parallel_image.astype(float)
    # crossed_image = crossed_image.astype(float)
    # biased_image = biased_image.astype(float)
    
    denominator = parallel_image - dark_image
    # set all 0 pixels to 1 to avoid division by zero
    denominator[denominator == 0] = 1
    
    transmission_image = (biased_image - crossed_image) / denominator
    # convert difference image to csv
    data = pd.DataFrame(transmission_image)
    csv = data.to_csv(index=False).encode('utf-8')

    # download button for the difference image
    st.download_button(
        label="Download transmission image",
        data=csv,
        file_name="transmission_image.csv",
        mime="text/csv",
    )
    image_min = transmission_image.min()
    image_max = transmission_image.max()

    min_val, max_val = st.slider(
        label="Color range slider:",
        min_value=-10.0,
        max_value=10.0,
        # value=[0, 500],
        step=0.1,
        key="transmission_slider"
    )
    
    figure5 = px.imshow(
        transmission_image, color_continuous_scale=color_theme, range_color=[min_val, max_val]
    )
    
    st.plotly_chart(figure5)