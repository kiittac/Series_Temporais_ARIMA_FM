import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Função para ajustar o modelo ARIMA e plotar o gráfico
def plotar_serie_temporal(df, titulo):
    serie_temporal = df['valores']
    
    # Ajustar o modelo ARIMA (ARIMA(1,1,1))
    modelo = ARIMA(serie_temporal, order=(1, 1, 1))
    resultado = modelo.fit()
    
    # Fazer previsões para os próximos 5 passos
    previsao = resultado.forecast(steps=5)
    
    # Plotar os dados originais
    plt.plot(serie_temporal, label='Série Temporal')
    
    # Adicionar as previsões no gráfico
    previsao_index = pd.date_range(start=serie_temporal.index[-1], periods=6, freq='D')[1:]
    plt.plot(previsao_index, previsao, label='Previsão', color='red')
    
    # Adicionar título e rótulos aos eixos
    plt.title(titulo)
    plt.xlabel('Data')
    plt.ylabel('Valor')
    
    # Exibir a legenda
    plt.legend()

# Carregar e plotar os dados do primeiro gráfico a partir do arquivo CSV
df_1_carregado = pd.read_csv('dados_1.csv', index_col='data', parse_dates=True)
plotar_serie_temporal(df_1_carregado, 'Série Temporal com Previsões ARIMA \nMantém o padrão de aumento de valor')
plt.show()

# Carregar e plotar os dados do segundo gráfico a partir do arquivo CSV
df_2_carregado = pd.read_csv('dados_2.csv', index_col='data', parse_dates=True)
plotar_serie_temporal(df_2_carregado, 'Série Temporal com Previsões ARIMA \nO valor aumenta e se mantém constante')
plt.show()