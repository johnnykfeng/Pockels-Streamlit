import streamlit as st
import numpy as np
import plotly.graph_objects as go
from malus_law_fit import (
    fit_malus_law,
    malus_law,
    radians_to_degrees,
    get_crossed_angle,
    get_parallel_angle,
)

st.set_page_config(
    page_title="Polarizer Calibration",
    page_icon=":bar_chart:",
)


st.title("Polarizer Calibration")
with st.expander("How to use: "):
    st.write("Instructions coming soon...")

with st.sidebar:
    st.header("-Fit parameters-")

# Get user input, give default values as an example
x_data = st.text_input(
    label="Enter angles in degrees (comma-separated):",
    value="0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160",
)
y_data = st.text_input(
    label="Enter intensity data (comma-separated):",
    value="0.75, 0.42, 0.34, 0.2, 0.24, 0.3, 0.68, 0.68, 0.87, 1.03, 1.3, 1.22, 1.33, 1.29, 0.97, 0.91, 0.84",
)

if not x_data or not y_data:
    st.warning("Please enter some data.")

if len(x_data.split(",")) != len(y_data.split(",")):
    st.warning(f"len(x) = {len(x_data.split(','))}")
    st.warning(f"len(y) = {len(y_data.split(','))}")
    st.warning("X and Y data must have the same number of elements.")

# Convert input to lists of floats
x_values = [float(x.strip()) for x in x_data.split(",")]
y_values = [float(y.strip()) for y in y_data.split(",")]
x_values = np.array(x_values)  # Convert to numpy array
y_values = np.array(y_values)

length_condition = len(x_values) == len(y_values)
none_condition = x_values is not None and y_values is not None
active_conditions = length_condition and none_condition

# initialize figure plot
fig = go.Figure()
fig.update_layout(
    xaxis_title="angle (degrees)", yaxis_title="Intensity", showlegend=False
)
st.session_state.fig = fig

def show_raw_data():

    # Plot the data
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers+lines",
            line=dict(color="orange", dash="dash"),
        )
    )
    fig.update_layout(
        showlegend=False,
    )
    st.session_state.fig = fig
    # st.plotly_chart(fig)

def do_curve_fit():
    popt, pcov = fit_malus_law(x_values, y_values)
    phi_deg = radians_to_degrees(popt[1])

    parallel_angle = get_parallel_angle(phi_deg)
    crossed_angle = get_crossed_angle(phi_deg)
    st.sidebar.write(f"**I0_fit:** {popt[0]:.2f}")
    st.sidebar.write(f"**phi_fit:** {phi_deg:.2f} degrees")
    st.sidebar.write(f"**offset_fit:** {popt[2]:.2f}")
    st.sidebar.write(
        f"**parallel_angle:** {parallel_angle:.2f}"
    )  # Display in sidebar
    st.sidebar.write(f"**crossed_angle:** {crossed_angle:.2f}")

    x_fit = np.arange(x_values[0], x_values[-1], 1)
    y_fit = malus_law(x_fit, *popt)

    # Plot the data
    fig.update_layout(
        showlegend=True,  # Add legend
    )
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="lines",
            name="curve fit",  # Add label for curve fit
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers+lines",
            line=dict(color="orange", dash="dash"),
            name="raw data",  # Add label for raw data
        )
    )
    st.session_state.fig = fig

def write_hello():
    st.write("Hello")

def write_bye():
    st.write("Good bye")

col1, col2, _, _ = st.columns(4)

with col1:
    if st.button(
        "Show Raw Data", disabled=not active_conditions, key="raw_data_button"
    ):
        show_raw_data()
with col2:
    if st.button(
        "Fit Malus Law", disabled=not active_conditions, key="curve_fit_button"
    ):
        do_curve_fit()

container = st.container(border=True)
# container.pyplot(st.session_state.fig)
container.plotly_chart(st.session_state.fig)

