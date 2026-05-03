import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your processed features
data = np.load('features.npy')

x = pd.DataFrame(data)

y = np.load('labels.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_test, y_predict)

print('{}% of samples were classified correctly !'.format(score * 100))

np.save('y_test.npy', y_test)
np.save('y_predict.npy', y_predict)

with open('sign_language_model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved successfully!")