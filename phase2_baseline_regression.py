from tensorflow import keras
from tensorflow.keras import layers
def build_regression_model(input_dim):
# TODO : instancier un modèle Sequential
# TODO : ajouter une couche Dense(64, activation='relu', input_shape=(input_dim,))
# TODO : ajouter une couche Dense(32, activation='relu')
# TODO : ajouter la couche de sortie : Dense(1) sans activation (régression = valeur
continue)
# TODO : compiler avec optimizer='adam', loss='mse', metrics=['mae']
# TODO : retourner le modèle
pass
model = build_regression_model(input_dim=8)
model.summary()
# TODO : appeler model.fit avec X_train_norm, y_train, epochs=100, batch_size=32,
# validation_data=(X_val_norm, y_val), verbose=1
# stocker le retour dans `history`
# TODO : appeler model.evaluate sur (X_test_norm, y_test, verbose=0)
# récupérer (test_loss, test_mae)
# afficher avec f"MAE test : {test_mae:.4f} (en centaines de milliers de $)"