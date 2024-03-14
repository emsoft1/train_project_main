from predict import predict, get_user_options, get_signal_for_plotting, get_vibration_dataframe

import streamlit as st
import matplotlib.pyplot as plt

import pandas as pd
import plotly.graph_objects as go
import numpy as np



ANIMATION_STEP = 6
NORMAL_COLOR = 'oldlace'
WARNING_COLOR = 'lightsalmon'
FAILPOINT = 550

def create_mean_vibration_animation(failpoint = None):

    df = get_vibration_dataframe()

    fig = go.Figure(
    data=[go.Scatter(x=np.arange(len(df.index[:ANIMATION_STEP])), y=df.iloc[:ANIMATION_STEP,0]),
          go.Scatter(x=np.arange(len(df.index[:ANIMATION_STEP])), y=df.iloc[:ANIMATION_STEP,1]),
          go.Scatter(x=np.arange(len(df.index[:ANIMATION_STEP])), y=df.iloc[:ANIMATION_STEP,2]),
          go.Scatter(x=np.arange(len(df.index[:ANIMATION_STEP])), y=df.iloc[:ANIMATION_STEP,3])
         ],
    layout=go.Layout(
        xaxis=dict(range=[0, len(df)], autorange=False),
        yaxis=dict(range=[0, 0.2], autorange=False),
        #title="Start Title",
        plot_bgcolor = NORMAL_COLOR,
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
    ),

    frames=[go.Frame( data=[go.Scatter(x=np.arange(len(df.index[:i*ANIMATION_STEP])), y=df.iloc[:i*ANIMATION_STEP,0]),
                            go.Scatter(x=np.arange(len(df.index[:i*ANIMATION_STEP])), y=df.iloc[:i*ANIMATION_STEP,1]),
                            go.Scatter(x=np.arange(len(df.index[:i*ANIMATION_STEP])), y=df.iloc[:i*ANIMATION_STEP,2]),
                            go.Scatter(x=np.arange(len(df.index[:i*ANIMATION_STEP])), y=df.iloc[:i*ANIMATION_STEP,3])
                           ],layout=go.Layout(plot_bgcolor = (WARNING_COLOR if (failpoint and failpoint <= i*ANIMATION_STEP) else NORMAL_COLOR))
                    ) for i in range(len(df)//ANIMATION_STEP + 2 ) ]
    )
    return fig




st.title('Anomaly Detection for Bearings Vibrations')


user_options = get_user_options()

option = st.selectbox(
    "Select a vibration snapshot to be analyzed!",
    user_options)

if st.button("Check for anomaly!"):
    if not option:
        st.warning("No timestamp selected!")
    else:
        #plot if desired
        if True:
            signal = get_signal_for_plotting(option)
            ae_signal = get_signal_for_plotting(option, autoencoded=True)
            fig, ax = plt.subplots(2,1)
            fig.set_size_inches((12,5.5))
            fig.tight_layout(pad=1.5)
            ax[0].plot(signal)
            ax[0].set_title('Original Signal', loc='center')
            ax[1].plot(ae_signal)
            ax[1].set_title('Autoencoded Signal', loc='center')
            st.pyplot(fig)

        if (predict(option)):
            st.error('Anomaly detected! Maintenance recommended!')
        else:
            st.success("Vibrations look normal. Carry on!")

if st.button("Show Signal!"):
    if not option:
        st.warning("No timestamp selected!")
    else:
        signal = get_signal_for_plotting(option)
        fig, ax = plt.subplots()
        ax.plot(signal)
        st.pyplot(fig)

if st.button("Plot averaged vibrations!"):
    fig = create_mean_vibration_animation()
    st.plotly_chart(fig)

if st.button("Monitor vibrations!"):
    fig = create_mean_vibration_animation(failpoint=FAILPOINT)
    st.plotly_chart(fig)
