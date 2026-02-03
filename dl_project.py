import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna


# 1. Generate Synthetic Dataset

def generate_synthetic_data(n_samples=2000):
    t = np.arange(n_samples)
    seasonal1 = np.sin(2 * np.pi * t / 24)  # daily seasonality
    seasonal2 = np.sin(2 * np.pi * t / 168) # weekly seasonality
    trend = 0.001 * t
    noise = np.random.normal(0, 0.1, n_samples)
    series = seasonal1 + seasonal2 + trend + noise
    df = pd.DataFrame({"value": series})
    return df

data = generate_synthetic_data()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


# 2. Prepare Time Series Windows

def create_dataset(series, window_size=48):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

window_size = 48
X, y = create_dataset(scaled_data, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]


# 3. Baseline LSTM Model

def build_lstm_model():
    model = models.Sequential([
        layers.LSTM(64, input_shape=(window_size, 1)),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

baseline_model = build_lstm_model()
baseline_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

baseline_preds = baseline_model.predict(X_test)
print("Baseline RMSE:", np.sqrt(mean_squared_error(y_test, baseline_preds)))
print("Baseline MAE:", mean_absolute_error(y_test, baseline_preds))


# 4. Attention Layer

class AttentionLayer(layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)


# 5. Attention-Augmented Model

def build_attention_model(trial):
    units = trial.suggest_int("units", 32, 128)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    inputs = layers.Input(shape=(window_size, 1))
    lstm_out = layers.LSTM(units, return_sequences=True)(inputs)
    att_out = AttentionLayer()(lstm_out)
    dense_out = layers.Dense(1)(att_out)

    model = models.Model(inputs=inputs, outputs=dense_out)
    model.compile(optimizer="adam", loss="mse")
    return model


# 6. Hyperparameter Optimization
 
def objective(trial):
    model = build_attention_model(trial)
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_params
print("Best Hyperparameters:", best_params)


# 7. Train Final Attention Model

final_model = build_attention_model(study.best_trial)
final_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

final_preds = final_model.predict(X_test)
print("Attention RMSE:", np.sqrt(mean_squared_error(y_test, final_preds)))
print("Attention MAE:", mean_absolute_error(y_test, final_preds))


# Visualize Attention Weights

attention_model = models.Model(inputs=final_model.input,
                               outputs=final_model.layers[2].output)

att_weights = attention_model.predict(X_test[:1])
plt.plot(att_weights[0])
plt.title("Attention Weights for First Test Sample")
plt.show()