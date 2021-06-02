import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


#carregamento da base de dados da usina
geracao = pd.read_csv('dados\Plant_1_Generation_Data.csv')
clima = pd.read_csv('dados\Plant_1_Weather_Sensor_Data.csv')
print(geracao.describe())