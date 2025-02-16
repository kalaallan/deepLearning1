import tensorflow as tf
import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


# charger mnist depuis openml 
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

#convertir les étiquettes en entier
y = y.astype(int)

# Afficher quelques images
'''fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Exemples d'images du dataset MNIST", fontsize=14)

for i, ax in enumerate(axes.flat):
    image = X[i].reshape(28, 28)  # Reshape en 28x28 pixels
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")

plt.show()
'''
# Normaliser les pixels (0 à 1)
X = X / 255.0

# 3Séparer en ensemble d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vérifier les dimensions
print(f"Train: {X_train.shape}, Labels: {y_train.shape}")
print(f"Test: {X_test.shape}, Labels: {y_test.shape}")

#création du model
model = Sequential([
    Flatten(input_shape=(784, )),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Entrainement du model
model.fit(X_train, y_train, epochs=5, batch_size=32)

#evaluer le model sur des données de tests
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Précision sur l'ensemble de test: {test_accuracy:.4f}")

#affichage des prédictions
y_pred = model.predict(X_test)

y_pred_classes = y_pred.argmax(axis=1)  # Prendre les classes avec la plus haute probabilité

# Calcul de la matrice de confusion
cm = confusion_matrix(y_test, y_pred_classes)

# Affichage avec seaborn (plus joli et facile à lire)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.title("Matrice de Confusion")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.show()
'''
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Exemples d'images du dataset MNIST", fontsize=14)

for i, ax in enumerate(axes.flat):
    image = X_test[i].reshape(28, 28)  # Reshape en 28x28 pixels
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Prédiction: {y_pred[i].argmax()}")
    ax.axis("off")

plt.show()
'''


