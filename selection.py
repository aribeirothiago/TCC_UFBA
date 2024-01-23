#%% IMPORTS

import pandas as pd
import pickle 
import datetime
from sklearn import preprocessing
from sklearn.utils import shuffle

now = datetime.datetime.now() # current date and time
date_time = now.strftime("%Y%m%d_%H%M%S")

#Importar base de dados

dfprod = pd.read_csv("data_prod.csv",sep=',', error_bad_lines=False)
dfprod = dfprod.replace(r'\\,','.', regex=True)
dfprod = dfprod.set_index('ZCGINSTAL')

dfprod= dfprod.drop('INSTPARCEIROFRAUDE', axis=1)

with open(r"../MODELOS/RNA/rna_20220714_104726.sav", "rb") as input_file:
   modelo = pickle.load(input_file)

#%% TRATAMENTO VALORES IMPOSSÍVEL E NULOS

dfprod['CORTEEFETIVO'][pd.isna(dfprod['CORTEEFETIVO'])] = 0
dfprod['RECORTEEFETIVO'][pd.isna(dfprod['RECORTEEFETIVO'])] = 0
dfprod['RECORTEEXECUTADO'][pd.isna(dfprod['RECORTEEXECUTADO'])] = 0
dfprod['PERDA_NAO_TECNICA_PRCT'][dfprod['PERDA_NAO_TECNICA_PRCT']<=0 ] = \
    dfprod['PERDA_NAO_TECNICA_PRCT'][dfprod['PERDA_NAO_TECNICA_PRCT']>=0].mean()
dfprod['PERDA_NAO_TECNICA_PRCT'][pd.isna(dfprod['PERDA_NAO_TECNICA_PRCT'])] =\
    dfprod['PERDA_NAO_TECNICA_PRCT'][dfprod['PERDA_NAO_TECNICA_PRCT']>=0].mean()
dfprod['QTD_FRAUDE'][pd.isna(dfprod['QTD_FRAUDE'])] = 0
dfprod['LATITUDE'][pd.isna(dfprod['LATITUDE'])] = dfprod['LATITUDE'].mean()
dfprod['LONGITUDE'][pd.isna(dfprod['LONGITUDE'])] = dfprod['LONGITUDE'].mean()

#%% CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS

le_motivo = preprocessing.LabelEncoder()
le_tip_instalacao = preprocessing.LabelEncoder()
le_fab_medidor = preprocessing.LabelEncoder()
le_modelo_medidor = preprocessing.LabelEncoder()
le_classe_contacontrato = preprocessing.LabelEncoder()
le_subclasse_contacontrato = preprocessing.LabelEncoder()
le_modliga = preprocessing.LabelEncoder()
le_ocle_atual = preprocessing.LabelEncoder()
le_familiainspant = preprocessing.LabelEncoder()
le_tipo_local = preprocessing.LabelEncoder()

dfprod['MOTIVO'] = le_motivo.fit_transform(dfprod['MOTIVO'])
dfprod['TIP_INSTALACAO'] = \
    le_tip_instalacao.fit_transform(dfprod['TIP_INSTALACAO'])
dfprod['FAB_MEDIDOR'] = le_fab_medidor.fit_transform(dfprod['FAB_MEDIDOR'])
dfprod['MODELO_MEDIDOR'] = \
    le_modelo_medidor.fit_transform(dfprod['MODELO_MEDIDOR'])
dfprod['CLASSE_CONTACONTRATO'] = \
    le_classe_contacontrato.fit_transform(dfprod['CLASSE_CONTACONTRATO'])
dfprod['SUBCLASSE_CONTACONTRATO'] = \
    le_subclasse_contacontrato.fit_transform(dfprod['SUBCLASSE_CONTACONTRATO'])
dfprod['MODLIGA'] = le_modliga.fit_transform(dfprod['MODLIGA'])
dfprod['OCLE_ATUAL'] = le_ocle_atual.fit_transform(dfprod['OCLE_ATUAL'])
dfprod['FAMILIAINSPANT'] = \
    le_familiainspant.fit_transform(dfprod['FAMILIAINSPANT'])
dfprod['TIPO_LOCAL'] = le_tipo_local.fit_transform(dfprod['TIPO_LOCAL'])

#%% RESULTADOS

dfprod = dfprod.dropna()

previsoes_prod = modelo.predict_proba(dfprod)
df_result = pd.DataFrame(previsoes_prod, index = dfprod.index)

df_result = shuffle(df_result)

df_result.to_csv('../RESULTADOS/resultado_'+str(modelo)+'_'+date_time+\
                 '.csv',sep=';')