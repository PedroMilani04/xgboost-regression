import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#------------------------------------------------------------####-------------------------------------------------------------------------#
# preencher dados faltantes e criar prorjecao_preenchido.csv

# Y
file_path = "./projecao.csv"
df = pd.read_csv(file_path)

# preencher o começo com backward fill (bfill) e o final com forward fill (ffill)
df['Y'] = df['Y'].fillna(method='bfill').fillna(method='ffill')
df.to_csv("./projecao_preenchido.csv", index=False)

#Xn
# optei por fazer o tratamento com a média específica de cada um e não uma iteração genérica usando a média da propria range, para ver como se sairía
df['X06'] = df['X06'].fillna(0.05) # média da range (usei a média da range em si pois os valores são muito pequenos, de 0 - 0.1)
df.to_csv("./projecao_preenchido.csv", index=False)

df['X12'] = df['X12'].fillna(6.3) # seguindo o histograma, média da range com maior incidência (27% dos valores: 5.5 - 7.1)
df.to_csv("./projecao_preenchido.csv", index=False)

df['X16'] = df['X16'].fillna(97.4) # seguindo o histograma, média da range com maior incidência (25% dos valores: 94.8 - 100.0)
df.to_csv("./projecao_preenchido.csv", index=False)

df['X17'] = df['X17'].fillna(12.3) # seguindo o histograma, média da range com maior incidência (17% dos valores: 11.5 - 13.1)
df.to_csv("./projecao_preenchido.csv", index=False)

df['X18'] = df['X18'].fillna(105.5) # seguindo o histograma, média da range com maior incidência (23% dos valores: 101.0 - 110.0)
df.to_csv("./projecao_preenchido.csv", index=False)

df['X19'] = df['X19'].fillna(229) # seguindo o histograma, média da range com maior incidência (22% dos valores: 216.0 - 242.0)
df.to_csv("./projecao_preenchido.csv", index=False)

df['X20'] = df['X20'].fillna(85) # seguindo o histograma, média da range com maior incidência (14% dos valores: 82.4 - 87.6)
df.to_csv("./projecao_preenchido.csv", index=False)

df['X21'] = df['X21'].fillna(83.1) # seguindo o histograma, média da range com maior incidência (22% dos valores: 80.2  - 86.0)
df.to_csv("./projecao_preenchido.csv", index=False)

#aqui, a range dos dados era muito grande e tinhamos um gap enorme de dados faltantes no inicio de cada coluna.
#para dar maior naturalidade aos dados, preenchi os dados faltantes de X22 e X23 com valores aleatórios de suas respectivas ranges. 
#O formato geral dos histogramas foi mantido, o que não aconteceu com o backward fill que testei antes.
df.loc[df["X22"].isna(), "X22"] = np.random.uniform(2600, 16900, df["X22"].isna().sum())
df.to_csv("./projecao_preenchido.csv", index=False)

df.loc[df["X23"].isna(), "X23"] = np.random.uniform(1500, 23800, df["X23"].isna().sum())
df.to_csv("./projecao_preenchido.csv", index=False)

#------------------------------------------------------------####-------------------------------------------------------------------------#
#tratar a correlação das colunas

corr_matrix = df.drop(columns=["DATA"]).corr()  # remover coluna de datas e faz a correlação

# Plotar um heatmap para visualização
#plt.figure(figsize=(12, 8))
#sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, linewidths=0.5)
#plt.title("Matriz de Correlação")
#plt.show()


# correlação alta
threshold = 0.9

colunas_para_remover = set()

# iteração na matriz de correlação
#(ela é simetrica, então x1 não precisa comparar com x0, só com os da frente, pois ja foi comparado por x0 em si)
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):  # evitar duplicação 
        if abs(corr_matrix.iloc[i, j]) > threshold:  # se a correlação for muito alta,
            colunas_para_remover.add(corr_matrix.columns[j])  # marca para remoção

print(colunas_para_remover)

df = df.drop(columns=colunas_para_remover)
df.to_csv("./projecao_preenchido_corrigido.csv", index=False)  # Salva após remover as colunas correlacionadas

#corr_matrix = df.drop(columns=["DATA"]).corr()  # Remover coluna de datas
#plt.figure(figsize=(12, 8))
#sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, linewidths=0.5)
#plt.title("Matriz de Correlação")
#plt.show()

#------------------------------------------------------------####-------------------------------------------------------------------------#
# previsão e estudo dos dados com XGBoost

file_path = "./projecao_preenchido_corrigido.csv"  # Usa o novo arquivo ao invés do original
df = pd.read_csv(file_path)

# convertendo a coluna de datas para datetime
df["DATA"] = pd.to_datetime(df["DATA"])

X = df.drop(columns=["Y", "DATA"])  # Remover 'Y' (alvo) e 'DATA' (não é uma feature numérica)
y = df["Y"]

X_train, X_test, y_train, y_test, data_train, data_test = train_test_split(X, y, df["DATA"], test_size=0.2, random_state=42)

model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=150, learning_rate=0.3, random_state=42)
model.fit(X_train, y_train)

# fazendo previsões
y_pred = model.predict(X_test)

# avaliando o modelo
mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred)
# na Learning Rate 0.1, os MSE tendem a ser sempre 620-660
print(f"erro Quadrático Médio (MSE): {mse:.2f}") # erro médio ao quiadrado entre os valores reais e da predição (570-765) (L.R: 0.3)
print(f"coeficiente de Determinação (R²): {r2:.2f}") # % das variaçoes dos dados explicadas pelo modelo (0.67 - 0.75) (L.R: 0.3)

# cria o DataFrame para visualização dos resultados
df_resultados = pd.DataFrame({"DATA": data_test, "Y_real": y_test, "Y_predito": y_pred})
df_resultados = df_resultados.sort_values(by="DATA")  # ordenar pela data

# potar os valores reais vs previstos ao longo do tempo
plt.figure(figsize=(12, 6))
plt.plot(df_resultados["DATA"], df_resultados["Y_real"], label="Valores Reais", color="blue", marker="o")
plt.plot(df_resultados["DATA"], df_resultados["Y_predito"], label="Valores Previstos", color="red", linestyle="dashed", marker="x")
plt.xlabel("Data")
plt.ylabel("Y")
plt.title("Comparação entre Valores Reais e Previstos ao longo do Tempo")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()