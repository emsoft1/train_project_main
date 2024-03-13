from predict import predict, get_user_options, get_signal_for_plotting, get_vibration_dataframe

import streamlit as st
import matplotlib.pyplot as plt

import pandas as pd
import plotly.graph_objects as go
import numpy as np


ANIMATION_STEP = 24
NORMAL_COLOR = 'lightcyan'
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
#     frames=[go.Frame(data=[go.Scatter(x=[1, 2], y=[1, 2])]),
#             go.Frame(data=[go.Scatter(x=[1, 4], y=[1, 4])] ),
#             go.Frame(data=[go.Scatter(x=[3, 4], y=[3, 4])],
#                      layout=go.Layout(title_text="End Title"))]


    frames=[go.Frame( data=[go.Scatter(x=np.arange(len(df.index[:i*ANIMATION_STEP])), y=df.iloc[:i*ANIMATION_STEP,0]),
                            go.Scatter(x=np.arange(len(df.index[:i*ANIMATION_STEP])), y=df.iloc[:i*ANIMATION_STEP,1]),
                            go.Scatter(x=np.arange(len(df.index[:i*ANIMATION_STEP])), y=df.iloc[:i*ANIMATION_STEP,2]),
                            go.Scatter(x=np.arange(len(df.index[:i*ANIMATION_STEP])), y=df.iloc[:i*ANIMATION_STEP,3])
                           ],layout=go.Layout(plot_bgcolor = (WARNING_COLOR if (failpoint and failpoint <= i*ANIMATION_STEP) else NORMAL_COLOR))
                    ) for i in range(len(df)//ANIMATION_STEP + 2 ) ]
    )
    return fig




st.title('Anomaly Detection for Predictive Maintenance of Bearings')

#st.text("Please select a timestamp!")

user_options = get_user_options()

option = st.selectbox(
    "Select a vibration snapshot to be analyzed!",
    user_options)

if st.button("Check for anomaly!"):
    if not option:
        st.warning("No timestamp selected!")
    else:
        if (predict(option)):
            st.error('Maintenance needed!')
        else:
            st.success("Vibrations look normal. Carry on!")

if st.button("Show Signal!"):
    if not option:
        st.warning("No timestamp selected!")
    else:
        signal = get_signal_for_plotting(option)
        fig, ax = plt.subplots()
        ax.plot(signal.transpose())
        st.pyplot(fig)

if st.button("Plot averaged vibrations!"):
    fig = create_mean_vibration_animation()
    st.plotly_chart(fig)

if st.button("Monitor vibrations!"):
    fig = create_mean_vibration_animation(failpoint=FAILPOINT)
    st.plotly_chart(fig)



# signal = get_signal_for_plotting(get_user_options()[0])
# print(signal, signal.shape)
# fig, ax = plt.subplots()
# ax.plot(signal.transpose())
# fig.savefig("test.png")


# option = st.selectbox(
#    "Select a vibration snapshot to be analyzed!",
#    user_options,
#    index=None
#    #,   placeholder="Select snapshot by timestamp..."
# )

# st.success('This is a success!')
# st.info('This is an info')
# st.warning('This is a semi success')
# st.error('Let\'s keep positive, this might be pretty close to a success!')
