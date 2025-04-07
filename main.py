import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

#------------------------------------------------------------#
# 📌 Carregar o dataset
df = pd.read_csv("./projecao.csv")

# 📌 Criar colunas "Mês" e "Ano"
df["DATA"] = pd.to_datetime(df["DATA"])
df["Mes"] = df["DATA"].dt.month
df["Ano"] = df["DATA"].dt.year

# 📌 Lista das colunas Xn com valores faltantes a serem preenchidos
colunas_com_nan = ["X06", "X12", "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23"]

# 📌 Preencher os Xn usando XGBoost
for coluna in colunas_com_nan:
    print(f"Treinando modelo para preencher {coluna}...")

    df_temp = df.drop(columns=['DATA', 'Y'])  # Remover DATA e Y para prever apenas Xn
    colunas_completas = df_temp.columns[df_temp.notna().all()].tolist()

    if colunas_completas:
        X = df_temp[colunas_completas]
        y = df[coluna]

        X_train = X[y.notna()]
        y_train = y[y.notna()]
        X_pred = X[y.isna()]

        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.3, random_state=42)
        model.fit(X_train, y_train)

        df.loc[y.isna(), coluna] = model.predict(X_pred)

# 📌 Salvar dataset com os Xn preenchidos
df.to_csv("./projecao_preenchido_corrigido.csv", index=False)
print("Preenchimento dos Xn finalizado e salvo!")

#------------------------------------------------------------#
# 📊 Previsão de Y usando os Xn preenchidos + "Mês" e "Ano"

df = pd.read_csv("./projecao_preenchido_corrigido.csv")

# 🔹 Recriar as colunas "Mês" e "Ano" após a leitura do CSV
df["DATA"] = pd.to_datetime(df["DATA"])
df["Mes"] = df["DATA"].dt.month
df["Ano"] = df["DATA"].dt.year

# 🔹 Selecionar features e alvo
X = df.drop(columns=["Y", "DATA"])
y = df["Y"]

# 🔹 Separar dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(X[y.notna()], y[y.notna()], test_size=0.3, random_state=42)

#------------------------------------------------------------#
# 🛠️ Otimização de Hiperparâmetros com Optuna

def objective(trial):
    """Função objetivo para otimizar os hiperparâmetros do XGBoost."""
    
    params = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "lambda": trial.suggest_float("lambda", 1, 10),
        "alpha": trial.suggest_float("alpha", 0, 10),
        "random_state": 42
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return mae  # Queremos minimizar o erro

# 🔹 Rodar Optuna para encontrar os melhores parâmetros
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# 📌 Melhor conjunto de hiperparâmetros encontrados
best_params = study.best_params
print("✅ Melhores parâmetros encontrados:", best_params)

#------------------------------------------------------------#
# 🔹 Avaliação final com K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
maes, mapes = [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = xgb.XGBRegressor(**best_params)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

    maes.append(mae)
    mapes.append(mape)
    print(f"🔹 Fold {fold+1}: MAE={mae:.4f}, MAPE={mape:.2f}%")

# 📢 Exibir médias finais
print(f"\n✅ Avaliação Final com K-Fold (5 Folds)")
print(f"🔹 MAE Médio: {np.mean(maes):.4f}")
print(f"🔹 MAPE Médio: {np.mean(mapes):.2f}%")

# 🔹 Previsão dos valores ausentes de Y
X_pred = X[y.isna()]
df.loc[y.isna(), "Y"] = model.predict(X_pred)

df.to_csv("./projecao_final.csv", index=False)
print("Previsão de Y concluída e salva!")

#------------------------------------------------------------#
# 📈 Visualização dos dados previstos

df_resultados = df[["DATA", "Y"]]
plt.figure(figsize=(12, 6))
plt.plot(df_resultados["DATA"], df_resultados["Y"], label="Valores Previstos de Y", color="red", linestyle="dashed", marker="x")
plt.xlabel("Data")
plt.ylabel("Y")
plt.title("Valores Previstos de Y ao longo do Tempo")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
