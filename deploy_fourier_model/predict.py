import joblib
import pandas as pd

THRESHOLD = 0.19
sample_data = pd.read_csv("ui_data_exp2_1024_buckets.csv")
user_options = sample_data.timestamp.to_list()
sample_data.set_index('timestamp', inplace=True)
model = joblib.load("scaler_and_ft_exp2_1024b.mod")
scaler = joblib.load("scaler.mod")

def get_user_options():
    return user_options

def predict(time_stamp):
    test_signal = sample_data.loc[time_stamp,:]
    signal_as_np = test_signal.to_numpy().reshape((1,-1))
    prediction = model.predict(signal_as_np)
    original_signal_scaled = scaler.transform(signal_as_np)

    loss = ((prediction - original_signal_scaled)**2).mean()

    if loss < THRESHOLD:
        return False
    return True

def get_signal_for_plotting(time_stamp):
    test_signal = sample_data.loc[time_stamp,:]
    signal_as_np = test_signal.to_numpy().reshape((1,-1))
    return signal_as_np

# if __name__ == '__main__':
#     out = []
#     for time in get_user_options():
#         out.append(predict(time))
#     print(out)
