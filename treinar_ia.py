import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Carrega os dados do arquivo CSV
df = pd.read_csv('alfabeto_libras.csv')

# X são as coordenadas (features), y é a letra (label)
X = df.drop('letra', axis=1)
y = df['letra']

# Divide os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Cria e treina o modelo RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("--- Iniciando o treinamento do modelo... ---")
model.fit(X_train, y_train)
print("--- Treinamento finalizado! ---")

# Avalia a acurácia do modelo no conjunto de teste
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo no conjunto de teste: {accuracy * 100:.2f}%")

# Salva o modelo treinado para uso posterior
with open('reconhecedor_libras.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo salvo com sucesso como 'reconhecedor_libras.pkl'.")