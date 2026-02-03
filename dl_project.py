import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna


def generate_synthetic_data(n_samples=2000):
    t = np.arange(n_samples)
    daily = np.sin(2 * np.pi * t / 24)
    weekly = np.sin(2 * np.pi * t / 168)
    trend = 0.001 * t
    noise = np.random.normal(0, 0.1, n_samples)
    series = daily + weekly + trend + noise
    return pd.DataFrame({"value": series})


data = generate_synthetic_data()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


def create_dataset(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)


WINDOW_SIZE = 48

X, y = create_dataset(scaled_data, WINDOW_SIZE)
X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


def build_lstm_model():
    model = models.Sequential([
        layers.LSTM(64, input_shape=(WINDOW_SIZE, 1)),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


baseline_model = build_lstm_model()
baseline_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

baseline_preds = baseline_model.predict(X_test)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
baseline_mae = mean_absolute_error(y_test, baseline_preds)

print("Baseline RMSE:", baseline_rmse)
print("Baseline MAE:", baseline_mae)


class AttentionLayer(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, inputs):
        score = self.V(tf.nn.tanh(self.W(inputs)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context, weights


def build_attention_model(trial):
    units = trial.suggest_int("units", 32, 128)

    inputs = layers.Input(shape=(WINDOW_SIZE, 1))
    lstm_out = layers.LSTM(units, return_sequences=True)(inputs)
    context, _ = AttentionLayer(units)(lstm_out)
    output = layers.Dense(1)(context)

    model = models.Model(inputs, output)
    model.compile(optimizer="adam", loss="mse")
    return model


def objective(trial):
    model = build_attention_model(trial)
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    preds = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, preds))


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best Hyperparameters:", study.best_params)


final_model = build_attention_model(study.best_trial)
final_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

final_preds = final_model.predict(X_test)
attention_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
attention_mae = mean_absolute_error(y_test, final_preds)

print("Attention RMSE:", attention_rmse)
print("Attention MAE:", attention_mae)


def rolling_window_evaluation(model_fn, X, y, train_size=400, step=100):
    rmses = []

    for start in range(0, len(X) - train_size - step, step):
        X_tr = X[start:start + train_size]
        y_tr = y[start:start + train_size]
        X_te = X[start + train_size:start + train_size + step]
        y_te = y[start + train_size:start + train_size + step]

        model = model_fn()
        model.fit(X_tr, y_tr, epochs=5, batch_size=32, verbose=0)
        preds = model.predict(X_te)
        rmses.append(np.sqrt(mean_squared_error(y_te, preds)))

    return np.mean(rmses), np.std(rmses)


rw_mean, rw_std = rolling_window_evaluation(build_lstm_model, X, y)
print("Rolling LSTM RMSE:", rw_mean, "+/-", rw_std)



attention_extractor = models.Model(
    inputs=final_model.input,
    outputs=final_model.layers[2].output[1]
)

weights = attention_extractor.predict(X_test[:50])
avg_weights = np.mean(weights, axis=0)

plt.plot(avg_weights.squeeze())
plt.xlabel("Time Step")
plt.ylabel("Average Attention Weight")
plt.title("Average Temporal Attention Across Test Samples")
plt.show()
