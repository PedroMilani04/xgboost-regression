import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv("projecao_preenchido2.csv")  # Ajuste o nome do arquivo se necessário

# Contar valores ausentes
missing_values = df.isna().sum()

# Contagem total de valores ausentes em Y
missing_y_count = missing_values["Y"]

# Contagem total de valores ausentes nas variáveis Xn
missing_xn = missing_values.drop(["DATA", "Y"])  # Removendo colunas DATA e Y

# Total de valores ausentes em Xn
missing_xn_total = missing_xn.sum()

# Exibir resultados
print("Valores ausentes em Y:", missing_y_count)
print("Valores ausentes em Xn:", missing_xn_total)
print("Detalhes das colunas Xn com valores ausentes:\n", missing_xn[missing_xn > 0])
