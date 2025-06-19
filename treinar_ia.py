import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Carregar os dados do arquivo CSV
df = pd.read_csv('alfabeto_libras.csv')

# 2. Preparar os Dados
# X são as coordenadas (features), y é a letra (label)
X = df.drop('letra', axis=1) # Pega todas as colunas, exceto a 'letra'
y = df['letra']             # Pega apenas a coluna 'letra'

# 3. Dividir os dados em Conjunto de Treino e Conjunto de Teste
# 80% dos dados para treinar, 20% para testar a performance do modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Criar e Treinar o Modelo
# Usaremos o RandomForest, um modelo robusto e eficiente
model = RandomForestClassifier(n_estimators=100, random_state=42)

print("--- Iniciando o treinamento do modelo... ---")
model.fit(X_train, y_train)
print("--- Treinamento finalizado! ---")

# 5. Avaliar a Performance do Modelo
# O modelo fará previsões no conjunto de teste, que ele nunca viu antes
y_pred = model.predict(X_test)

# Comparamos as previsões com as respostas corretas para calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo no conjunto de teste: {accuracy * 100:.2f}%")

# 6. Salvar o Modelo Treinado
# Salvamos o "cérebro" do nosso modelo em um arquivo para usá-lo depois
with open('reconhecedor_libras.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo salvo com sucesso como 'reconhecedor_libras.pkl'")