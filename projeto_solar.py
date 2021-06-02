# -*- coding: utf-8 -*-
"""projeto_solar.ipynb

Autor: Fillipe de Almeida Andrade
"""

# Commented out IPython magic to ensure Python compatibility.
#importação das bibliotecas iniciais
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
# %matplotlib inline

#carregamento da base de dados da usina
geracao = pd.read_csv('dados/Plant_1_Generation_Data.csv')
clima = pd.read_csv('dados/Plant_1_Weather_Sensor_Data.csv')

#escopo geral da base de dados
geracao.head()
clima.head()

#descrição do comportamento das varáveis dos dados
geracao.describe()
clima.describe()

a = geracao.isnull().sum()
b = clima.isnull().sum()
print(f'Quantidade de elemesntos nulo da geração:\n{a}')
print(f'Quantidade de elemesntos nulo ddo clima:\n{b}')

#Convertendo as datas para DateTime
geracao['DATE_TIME'] = pd.to_datetime(geracao['DATE_TIME'],format = '%d-%m-%Y %H:%M')
clima['DATE_TIME'] = pd.to_datetime(clima['DATE_TIME'],format = '%Y-%m-%d %H:%M')

#criando colunas para tempo e data
geracao['DATE'] = geracao['DATE_TIME'].apply(lambda x:x.date())
geracao['TIME'] = geracao['DATE_TIME'].apply(lambda x:x.time())
clima['DATE'] = clima['DATE_TIME'].apply(lambda x:x.date())
clima['TIME'] = clima['DATE_TIME'].apply(lambda x:x.time())

geracao.tail()
clima.tail()

#GeraçãodosMódulosFV
geracao_dia = geracao.copy()
geracao_dia = geracao_dia.groupby(['TIME','SOURCE_KEY'])['DAILY_YIELD'].mean().unstack()
plt.figure(figsize=(10,5))
geracao_dia.iloc[:,0:1].plot()
plt.title('Potência DC em um inversor Usina 1')
plt.ylabel('kWh')
plt.xlabel('Tempo')

conv_Inv= geracao.groupby(['SOURCE_KEY']).mean()
eficiencia= conv_Inv['AC_POWER']*1000/conv_Inv['DC_POWER']
eficiencia.plot(figsize=(15,5), style='o--')
plt.axhline(eficiencia.mean(),linestyle='--',color='green')
plt.title('Eficiência dos Inversores', size=20)
plt.ylabel('Eficiência (%)')
plt.xlabel('ID dos inversores')

#potencia CC gerado pelos módulos
geracao_cc = geracao.copy()
geracao_cc = geracao_cc.groupby(['TIME','DATE'])['DC_POWER'].sum().unstack()

fig,ax=plt.subplots(ncols=2,nrows=1,dpi=200,figsize=(20,5))
ax[0].set_title('Potência DC em um inversor A da Usina 1')
ax[0].set_ylabel('kW')
ax[0].set_xlabel('Tempo')
ax[1].set_title('Potência DC em um inversor B da Usina 1')
ax[1].set_ylabel('kW')
ax[1].set_xlabel('Tempo')
geracao_cc.iloc[:,0:1].plot(ax=ax[0],linewidth = 5)
geracao_cc.iloc[:,1:2].plot(ax=ax[1],linewidth = 5,color='orange')

#potencia AC convertido pelo inversor
geracao_ac = geracao.copy()
geracao_ac = geracao_ac.groupby(['TIME','DATE'])['AC_POWER'].sum().unstack()

fig,ax=plt.subplots(ncols=2,nrows=1,dpi=200,figsize=(20,5))
ax[0].set_title('Potência AC em um inversor A da Usina 1')
ax[0].set_ylabel('kW')
ax[0].set_xlabel('Tempo')
ax[1].set_title('Potência AC em um inversor B da Usina 1')
ax[1].set_ylabel('kW')
ax[1].set_xlabel('Tempo')
geracao_ac.iloc[:,0:1].plot(ax=ax[0],linewidth = 5)
geracao_ac.iloc[:,1:2].plot(ax=ax[1],linewidth = 5,color='orange')

#Agrupando os dados pela data
geracao_diaria = geracao.groupby(['DATE_TIME'],as_index=False).sum()
geracao_diaria.head()

#selecionando as variaveis de estudo
geracao_select = geracao_diaria[['DATE_TIME','DC_POWER','AC_POWER','DAILY_YIELD']]
geracao_select[45:50]

#drop da chave id da usina e do inversor que serão insgnificantes para a predição
clima_drop = clima.drop(['PLANT_ID', 'SOURCE_KEY'], axis=1)
clima_drop.head()

#juntando dados de geração e clima
usine = pd.merge(geracao_select,clima_drop, how='inner', on='DATE_TIME')
usine_no_time = usine.drop(['DATE','TIME'],axis =1)
usine_no_time.head()

#insight da relação entre as variaveis
sns.pairplot(usine[['DC_POWER','AC_POWER','DAILY_YIELD','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION']])

#observando o comportamento das variaveis de clima e da geração dos módulos FV
usine_clima = usine.copy()
clima_cc = usine_clima.groupby(['TIME']).mean()

fig,ax=plt.subplots(ncols=2,nrows=2,dpi=200,figsize=(15,5))
clima_cc['IRRADIATION'].plot(ax=ax[0,0])
clima_cc['AMBIENT_TEMPERATURE'].plot(ax=ax[0,1])
clima_cc['MODULE_TEMPERATURE'].plot(ax=ax[1,0])
clima_cc['DC_POWER'].plot(ax=ax[1,1])

ax[0,0].set_ylabel('IRRADIATION')
ax[0,1].set_ylabel('AMBIENT TEMPERATURE')
ax[1,0].set_ylabel('MODULE TEMPERATURE')
ax[1,1].set_ylabel('DC POWER')

#Correlação entre as variaveis da usina para a escolha da mais apropriada para geração DC
usine_no_time.columns = ['DATE_TIME','DC_POWER','AC_POWER','DAILY_YIELD','AMBIENT','MODULE','IRRADIATION']
one_correlation = usine_no_time[['DC_POWER','AC_POWER','DAILY_YIELD','AMBIENT','MODULE','IRRADIATION']]
corr = one_correlation.corr()

fig_dims = (2, 2) 
sns.heatmap(round(corr,2), annot=True, mask=(np.triu(corr,+1)))
plt.savefig('correla.png',format = 'png')

#após a escolha das variaveis com maior correlação, separação final da base de dados
base = usine[['DC_POWER','MODULE_TEMPERATURE','IRRADIATION']]
base.describe()
resultados = usine[['DC_POWER','DATE_TIME']]

#importação das bibliotecas da Rede Neural
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#Selecionando a base de treinamento e teste
#sendo 70% para treinamento e 30% teste
base_treinamento = base[0:int(0.7*len(base))]
base_teste = base[int(0.7*len(base)):]
base_teste

#normalizando a base de dados e teste
normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)
normalizador_previsao = MinMaxScaler(feature_range=(0,1))
normalizador_previsao.fit_transform( base_treinamento.iloc[:,0:1])

#criando listas dos atributos previsores e os valores reais para a RNA
previsores = []
real_dc = []
len(base_treinamento)

#preenchendo essas listas
for i in range(100,len(base_treinamento)):
    previsores.append(base_treinamento_normalizada[i-100:i,0:3])
    real_dc.append(base_treinamento_normalizada[i,0])
len(real_dc)

#trasnformando em array
previsores,real_dc = np.array(previsores),np.array(real_dc)
len(real_dc)

#RNA LSTM COM DROPOUT PARA O OVERFITING
regressor = Sequential()
regressor.add(LSTM(units=100,return_sequences=True, input_shape =
                   (previsores.shape[1],3)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1,activation='sigmoid'))

regressor.compile(optimizer='adam',loss='mean_squared_error',
                  metrics = ['mean_absolute_error'])

es = EarlyStopping(monitor = 'loss', min_delta= 1e-10, patience=10, verbose = 1)
rlr = ReduceLROnPlateau(monitor='loss', factor = 0.2, patience=5, verbose=1)
mcp = ModelCheckpoint(filepath= 'pesos.h5', monitor='loss', save_best_only=True,
                      verbose=1)

#TREINAMENTO
regressor.fit(previsores,real_dc,epochs=100,batch_size=32,
              callbacks=[es,rlr,mcp])

#Valores reais de teste
real_dc_teste = base_teste.iloc[:,0:1].values

#previsores de teste
dado_um = base[len(base) - len(base_teste) - 100:].values
dado_um = normalizador.transform(dado_um)

#preenchendo os valores de teste numa lista
X_teste = []
for v in range(100,len(dado_um)):
    X_teste.append(dado_um[v-100:v,0:6])
X_teste = np.array(X_teste)

#prevendo os valores e transformando a normalização para os numeros reais
previsoes = regressor.predict(X_teste)
previsoes = normalizador_previsao.inverse_transform(previsoes)

#Previsão da RNA
plt.plot(real_dc_teste,color='red',label = 'Potência Real')
plt.plot(previsoes,color='blue',label = 'Previsão')
plt.xlabel('Tempo')
plt.ylabel('Potência kW')
plt.legend()
plt.show()

#Fazendo Previsão com FB prophet
from fbprophet import Prophet
pred_gen_all=usine.copy()
#agrupando e renomeando data e potência
pred_gen_all=pred_gen_all.groupby('DATE_TIME')['DC_POWER'].sum().reset_index()
pred_gen_all.rename(columns={'DATE_TIME':'ds','DC_POWER':'y'},inplace=True)
#plot da base de teste
pred_gen=pred_gen_all[pred_gen_all['ds']<'2020-06-08'].copy()
pred_gen_t=pred_gen_all[pred_gen_all['ds']>'2020-06-08'].copy()
pred_gen.plot(x='ds',y='y',figsize=(17,5))
plt.legend('')
plt.title('DC_POWER',size=17)
plt.show()

#fatores que o Prophet está levando em consideração da rede Neural
m = Prophet(yearly_seasonality=False,daily_seasonality=True)
#treinamento do FB prophet
m.fit(pred_gen)
future =m.make_future_dataframe(periods=10*24*4,freq='15min')
forecast = m.predict(future)
plt.plot(forecast.set_index('ds')['yhat'],label="prediction")
plt.legend()

#parametrizando para plotar com a RNA
v = forecast.set_index('ds')['yhat']
vf = v[int(0.70*len(v)):]

date = usine['DATE'].to_list()
date_teste = date[int(0.7*len(base)):]
date_teste = pd.DataFrame(date_teste)

date = usine['DATE'].to_list()
date_treinamento = date[0:int(0.7*len(base))]
date_treinamento = pd.DataFrame(date_treinamento)

#ultmos ajustes antes do gráfico
df = pd.DataFrame(previsoes)
df.columns = ["DC_POWER"]
df = df.assign(DATE = date_teste)
df_real = pd.DataFrame(real_dc_teste)
df_real.columns = ["DC_POWER"]
df_real = df_real.assign(DATA = date_teste)
df_treinamento = pd.DataFrame(real_dc)
df_treinamento = df_treinamento.assign(DATA = date_treinamento)

#gráfico comparativo entra a RNA E O Facebook Prophet
fig,ax=plt.subplots(ncols=1,nrows=2,dpi=200,figsize=(9,3))
df_real['DC_POWER'].plot(ax=ax[0], color = 'red', linewidth=2,label = 'Potência Real')
df['DC_POWER'].plot(ax=ax[0],linewidth=2, label =  'Previsão')
pred_gen_t.plot(ax = ax[1], x='ds',y='y',label='Potência Real',color = 'red')
vf.plot(ax=ax[1],linewidth=2, label = 'Previsão')
ax[0].axes.get_xaxis().set_visible(False)
ax[0].set_title('Rede Neural LSTM')
ax[1].set_title('Facebook Prophet')
ax[0].set_ylabel('Potência kW')
ax[1].set_ylabel('Potência kW')
ax[1].set_xlabel('Tempo em Dias')
plt.legend()

#Precissão
from sklearn.metrics import r2_score,mean_absolute_error

precisaoRNA = r2_score(real_dc_teste,previsoes)
precisaoFace = r2_score(vf,previsoes)

print(f'Precisão da RNA:{round(100*precisaoRNA,2)}%\n')
print(f'Precisão do Prophet:{round(100*precisaoFace,2)}%\n')
