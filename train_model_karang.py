import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# data dummy: 3 fitur (warna, tekstur, bentuk)
X = np.array([
    [0.2, 0.5, 0.7],  # Acropora
    [0.4, 0.7, 0.6],  # Porites
    [0.6, 0.3, 0.4],  # Montipora
    [0.3, 0.6, 0.5],  # Acropora
    [0.5, 0.8, 0.7],  # Porites
])
y = np.array([0, 1, 2, 0, 1])  # label: 0=Acropora, 1=Porites, 2=Montipora

# buat model
model = RandomForestClassifier()
model.fit(X, y)

# simpan model + label
with open('model_karang.pkl', 'wb') as f:
    pickle.dump((model, ['Acropora', 'Porites', 'Montipora']), f)

print("Model karang tersimpan sebagai model_karang.pkl")
