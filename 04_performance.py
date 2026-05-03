import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Load the data from the other file
y_test = np.load('y_test.npy')
y_predict = np.load('y_predict.npy')

# 2. Create the Confusion Matrix
cm = confusion_matrix(y_test, y_predict)

# 3. Plot using Seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Sign')
plt.ylabel('Actual Sign')
plt.title('Sign Language Detection - Confusion Matrix')
plt.show()