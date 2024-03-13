from predict import predict, get_user_options, get_signal_for_plotting

import streamlit as st
import matplotlib.pyplot as plt

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
