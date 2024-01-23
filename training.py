#%% IMPORTS

import pandas as pd
import pickle 
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from opfython.models import SupervisedOPF
from yellowbrick.classifier import ConfusionMatrix
from sklearn import metrics
from sklearn.model_selection import cross_validate

now = datetime.datetime.now() # current date and time
date_time = now.strftime("%Y%m%d_%H%M%S")

#Importar base de dados

data_ref = datetime.datetime(2022, 8, 12)
df = pd.read_csv("data.csv",sep=',', error_bad_lines=False)

df = df.replace(r'\\,','.', regex=True)

df['DT_ULT_EXEC'] = pd.to_datetime(df['DT_ULT_EXEC'])
df['PERDA_NAO_TECNICA_PRCT'] = pd.to_numeric(df['PERDA_NAO_TECNICA_PRCT'])
df['LATITUDE'] = pd.to_numeric(df['LATITUDE'])
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'])  

#%% TRATAMENTO VALORES IMPOSSÍVEL E NULOS

df['CORTEEFETIVO'][pd.isna(df['CORTEEFETIVO'])] = 0
df['CORTEEXECUTADO'][pd.isna(df['CORTEEXECUTADO'])] = 0
df['RECORTEEFETIVO'][pd.isna(df['RECORTEEFETIVO'])] = 0
df['RECORTEEXECUTADO'][pd.isna(df['RECORTEEXECUTADO'])] = 0
df['RELIGACAORECORTE'][pd.isna(df['RELIGACAORECORTE'])] = 0
df['RELIGACAOEFETIVO'][pd.isna(df['RELIGACAOEFETIVO'])] = 0
df['RELIGACAOCORTE'][pd.isna(df['RELIGACAOCORTE'])] = 0
df['PERDA_NAO_TECNICA_PRCT'][df['PERDA_NAO_TECNICA_PRCT']<=0 ] = \
    df['PERDA_NAO_TECNICA_PRCT'][df['PERDA_NAO_TECNICA_PRCT']>=0].mean()
df['PERDA_NAO_TECNICA_PRCT'][pd.isna(df['PERDA_NAO_TECNICA_PRCT'])] = \
    df['PERDA_NAO_TECNICA_PRCT'][df['PERDA_NAO_TECNICA_PRCT']>=0].mean()
df['QTD_FRAUDE'][pd.isna(df['QTD_FRAUDE'])] = 0
df['LATITUDE'][pd.isna(df['LATITUDE'])] = df['LATITUDE'].mean()
df['LONGITUDE'][pd.isna(df['LONGITUDE'])] = df['LONGITUDE'].mean()

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

df['MOTIVO'] = le_motivo.fit_transform(df['MOTIVO'])
df['TIP_INSTALACAO'] = le_tip_instalacao.fit_transform(df['TIP_INSTALACAO'])
df['FAB_MEDIDOR'] = le_fab_medidor.fit_transform(df['FAB_MEDIDOR'])
df['MODELO_MEDIDOR'] = le_modelo_medidor.fit_transform(df['MODELO_MEDIDOR'])
df['CLASSE_CONTACONTRATO'] = \
    le_classe_contacontrato.fit_transform(df['CLASSE_CONTACONTRATO'])
df['SUBCLASSE_CONTACONTRATO'] = \
    le_subclasse_contacontrato.fit_transform(df['SUBCLASSE_CONTACONTRATO'])
df['MODLIGA'] = le_modliga.fit_transform(df['MODLIGA'])
df['OCLE_ATUAL'] = le_ocle_atual.fit_transform(df['OCLE_ATUAL'])
df['FAMILIAINSPANT'] = le_familiainspant.fit_transform(df['FAMILIAINSPANT'])
df['TIPO_LOCAL'] = le_tipo_local.fit_transform(df['TIPO_LOCAL'])

#%% DATAFRAME DE CONSUMO

dfcons = df.loc[:,'MES-01_CONS':'MES-60_CONS'] 
dfcons['DT_ULT_EXEC'] = df['DT_ULT_EXEC'] 
dfcons.set_index('DT_ULT_EXEC', inplace=True)

datas_exec = dfcons.index.tolist()
for i in range(0, len(df)):
    data_exec = datas_exec[i]
    difmes = (data_ref.year - data_exec.year) * 12 + \
        (data_ref.month - data_exec.month)    
    nova_row = dfcons.iloc[i].shift(periods=-(difmes+1))
    dfcons.iloc[i] = nova_row

dfcons.reset_index(inplace=True)

dfcons.drop('DT_ULT_EXEC', axis=1, inplace=True)
dfcons.drop(dfcons.loc[:,'MES-37_CONS':'MES-60_CONS'] , \
            inplace = True, axis = 1)
dfcons = dfcons.loc[:,'MES-01_CONS':'MES-36_CONS'].dropna()

#%% DATAFRAME DO MODELO

dfmod = pd.DataFrame()

dfmod['CONS_MES_FRAUDE'] = dfcons['MES-01_CONS']							
dfmod['LATITUDE'] = df['LATITUDE']
dfmod['LONGITUDE'] = df['LONGITUDE']
dfmod['MOTIVO'] = df['MOTIVO']
dfmod['TIP_INSTALACAO'] = df['TIP_INSTALACAO']
dfmod['FAB_MEDIDOR'] = df['FAB_MEDIDOR']
dfmod['MODELO_MEDIDOR'] = df['MODELO_MEDIDOR']
dfmod['IDADE_ATIVA_MEDIDOR'] = df['IDADE_ATIVA_MEDIDOR']
dfmod['IDADE_INSTALACAO'] = df['IDADE_INSTALACAO']
dfmod['CLASSE_CONTACONTRATO'] = df['CLASSE_CONTACONTRATO']
dfmod['SUBCLASSE_CONTACONTRATO'] = df['SUBCLASSE_CONTACONTRATO']
dfmod['TENSAOFORNEC'] = df['TENSAOFORNEC']
dfmod['TENSAOMED'] = df['TENSAOMED']
dfmod['MODLIGA'] = df['MODLIGA']
dfmod['OCLE_ATUAL'] = df['OCLE_ATUAL']
dfmod['FAMILIAINSPANT'] = df['FAMILIAINSPANT']
dfmod['TIPO_LOCAL'] = df['TIPO_LOCAL']
dfmod['BAIXARENDA'] = df['BAIXARENDA']
dfmod['INSTPARCEIROFRAUDE'] = df['INSTPARCEIROFRAUDE']
dfmod['FATPAGVENC_12M'] = df['FATPAGVENC_12M']
dfmod['MEDIA_ATRASO_12M'] = df['MEDIA_ATRASO_12M']
dfmod['MEDIA_PAGDIAS_12M'] = df['MEDIA_PAGDIAS_12M']
dfmod['MAXIMO_VENCIMENTO'] = df['MAXIMO_VENCIMENTO']
dfmod['MINIMA_VENCIMENTO'] = df['MINIMA_VENCIMENTO']
dfmod['QUEDA'] = df['QUEDA']
dfmod['TX_QUEDA_POS'] = df['TX_QUEDA_POS']
dfmod['TX_QUEDA_REC'] = df['TX_QUEDA_REC']
dfmod['QTD_QUEDA_12M'] = df['QTD_QUEDA_12M']
dfmod['QTD_QUEDA_24M'] = df['QTD_QUEDA_24M']
dfmod['QTD_QUEDA_36M'] = df['QTD_QUEDA_36M']
dfmod['QTD_QUEDA_48M'] = df['QTD_QUEDA_48M']
dfmod['QTD_QUEDA_60M'] = df['QTD_QUEDA_60M']
dfmod['COMP_MED12M_MEDVZ12M'] = df['COMP_MED12M_MEDVZ12M']
dfmod['COMP_MED24M_MEDVZ24M'] = df['COMP_MED24M_MEDVZ24M']
dfmod['COMP_MED36M_MEDVZ36M'] = df['COMP_MED36M_MEDVZ36M']
dfmod['COMP_CV12M_CVVZ12M'] = df['COMP_CV12M_CVVZ12M']
dfmod['COMP_CV24M_CVVZ24M'] = df['COMP_CV24M_CVVZ24M']
dfmod['COMP_CV36M_CVVZ36M'] = df['COMP_CV36M_CVVZ36M']
dfmod['COMP_MEDIA12_MED13_24M'] = df['COMP_MEDIA12_MED13_24M']
dfmod['COMP_MEDIA12_MED25_36M'] = df['COMP_MEDIA12_MED25_36M']
dfmod['CORTEEFETIVO'] = df['CORTEEFETIVO']
dfmod['CORTEEXECUTADO'] = df['CORTEEXECUTADO']
dfmod['RECORTEEFETIVO'] = df['RECORTEEFETIVO']
dfmod['RECORTEEXECUTADO'] = df['RECORTEEXECUTADO']
dfmod['RELIGACAORECORTE'] = df['RELIGACAORECORTE']
dfmod['RELIGACAOEFETIVO'] = df['RELIGACAOEFETIVO']
dfmod['RELIGACAOCORTE'] = df['RELIGACAOCORTE']
dfmod['PERDA_NAO_TECNICA_PRCT'] = df['PERDA_NAO_TECNICA_PRCT']
dfmod['QTD_FRAUDE'] = df['QTD_FRAUDE']
dfmod['FRAUDE'] = df['FRAUDE']

dfmod = dfmod.dropna()

#%% MATRIZ DE CORRELAÇÃO

dfmat = pd.DataFrame() #Não entram variáveis categóricas

dfmat['CONS_MES_FRAUDE'] = dfcons['MES-01_CONS']
dfmat['IDADE_ATIVA_MEDIDOR'] = dfmod['IDADE_ATIVA_MEDIDOR']
dfmat['IDADE_INSTALACAO'] = dfmod['IDADE_INSTALACAO']
dfmat['FATPAGVENC_12M'] = dfmod['FATPAGVENC_12M']
dfmat['MEDIA_ATRASO_12M'] = dfmod['MEDIA_ATRASO_12M']
dfmat['MEDIA_PAGDIAS_12M'] = dfmod['MEDIA_PAGDIAS_12M']
dfmat['MAXIMO_VENCIMENTO'] = dfmod['MAXIMO_VENCIMENTO']
dfmat['MINIMA_VENCIMENTO'] = dfmod['MINIMA_VENCIMENTO']
dfmat['QUEDA'] = dfmod['QUEDA']
dfmat['TX_QUEDA_POS'] = dfmod['TX_QUEDA_POS']
dfmat['TX_QUEDA_REC'] = dfmod['TX_QUEDA_REC']
dfmat['QTD_QUEDA_12M'] = dfmod['QTD_QUEDA_12M']
dfmat['QTD_QUEDA_24M'] = dfmod['QTD_QUEDA_24M']
dfmat['QTD_QUEDA_36M'] = dfmod['QTD_QUEDA_36M']
dfmat['QTD_QUEDA_48M'] = dfmod['QTD_QUEDA_48M']
dfmat['QTD_QUEDA_60M'] = dfmod['QTD_QUEDA_60M']
dfmat['COMP_MED12M_MEDVZ12M'] = dfmod['COMP_MED12M_MEDVZ12M']
dfmat['COMP_MED24M_MEDVZ24M'] = dfmod['COMP_MED24M_MEDVZ24M']
dfmat['COMP_MED36M_MEDVZ36M'] = dfmod['COMP_MED36M_MEDVZ36M']
dfmat['COMP_CV12M_CVVZ12M'] = dfmod['COMP_CV12M_CVVZ12M']
dfmat['COMP_CV24M_CVVZ24M'] = dfmod['COMP_CV24M_CVVZ24M']
dfmat['COMP_CV36M_CVVZ36M'] = dfmod['COMP_CV36M_CVVZ36M']
dfmat['COMP_MEDIA12_MED13_24M'] = dfmod['COMP_MEDIA12_MED13_24M']
dfmat['COMP_MEDIA12_MED25_36M'] = dfmod['COMP_MEDIA12_MED25_36M']
dfmat['CORTEEFETIVO'] = dfmod['CORTEEFETIVO']
dfmat['CORTEEXECUTADO'] = dfmod['CORTEEXECUTADO']
dfmat['RECORTEEFETIVO'] = dfmod['RECORTEEFETIVO']
dfmat['RECORTEEXECUTADO'] = dfmod['RECORTEEXECUTADO']
dfmat['RELIGACAORECORTE'] = dfmod['RELIGACAORECORTE']
dfmat['RELIGACAOEFETIVO'] = dfmod['RELIGACAOEFETIVO']
dfmat['RELIGACAOCORTE'] = dfmod['RELIGACAOCORTE']
dfmat['PERDA_NAO_TECNICA_PRCT'] = dfmod['PERDA_NAO_TECNICA_PRCT']
dfmat['QTD_FRAUDE'] = dfmod['QTD_FRAUDE']
dfmat['FRAUDE'] = dfmod['FRAUDE']

correlacoes = dfmat.corr()

f1 = plt.figure(figsize=(19, 15))
plt.matshow(dfmat.corr(), fignum=f1.number, cmap='coolwarm')
plt.xticks(range(dfmat.shape[1]), dfmat.columns, fontsize=14, rotation=90)
plt.yticks(range(dfmat.shape[1]), dfmat.columns, fontsize=14)
cb = plt.colorbar( )
cb.ax.tick_params(labelsize=14)
plt.title('Matriz de Correlação', fontsize=16);

#%%REMOÇÃO DE VARIÁVEIS CORRELATAS

dfmod = dfmod.drop('MAXIMO_VENCIMENTO', axis=1)
dfmod = dfmod.drop('QTD_QUEDA_24M', axis=1)
dfmod = dfmod.drop('QTD_QUEDA_36M', axis=1)
dfmod = dfmod.drop('QTD_QUEDA_48M', axis=1)
dfmod = dfmod.drop('COMP_CV24M_CVVZ24M', axis=1)
dfmod = dfmod.drop('CORTEEXECUTADO', axis=1)
dfmod = dfmod.drop('RELIGACAORECORTE', axis=1)
dfmod = dfmod.drop('RELIGACAOEFETIVO', axis=1)
dfmod = dfmod.drop('RELIGACAOCORTE', axis=1)

dfmat = dfmat.drop('MAXIMO_VENCIMENTO', axis=1)
dfmat = dfmat.drop('QTD_QUEDA_24M', axis=1)
dfmat = dfmat.drop('QTD_QUEDA_36M', axis=1)
dfmat = dfmat.drop('QTD_QUEDA_48M', axis=1)
dfmat = dfmat.drop('COMP_CV24M_CVVZ24M', axis=1)
dfmat = dfmat.drop('CORTEEXECUTADO', axis=1)
dfmat = dfmat.drop('RELIGACAORECORTE', axis=1)
dfmat = dfmat.drop('RELIGACAOEFETIVO', axis=1)
dfmat = dfmat.drop('RELIGACAOCORTE', axis=1)

f2 = plt.figure(figsize=(19, 15))
plt.matshow(dfmat.corr(), fignum=f2.number, cmap='coolwarm')
plt.xticks(range(dfmat.shape[1]), dfmat.columns, fontsize=14, rotation=90)
plt.yticks(range(dfmat.shape[1]), dfmat.columns, fontsize=14)
cb = plt.colorbar( )
cb.ax.tick_params(labelsize=14)
plt.title('Matriz de Correlação Reduzida', fontsize=16);

#%% MACHINE LEARNING

X_mod = dfmod.iloc[:,0:(len(dfmod.columns)-1)].values
y_mod = dfmod.iloc[:,(len(dfmod.columns)-1)].values

X_mod_train, X_mod_test, y_mod_train, y_mod_test = \
    model_selection.train_test_split(X_mod,y_mod, 
    test_size = 0.25, random_state = 1, shuffle=True)

#%% SVM

svm = SVC(kernel='linear', random_state = 1, C = 5)
svm.fit(X_mod_train, y_mod_train)

pickle.dump(svm, open('../MODELOS/SVM/svm_'+date_time+'.sav', 'wb'))

previsoes = svm.predict(X_mod_test)

f3 = plt.figure(figsize=(19, 15))
cm = ConfusionMatrix(svm)
cm.fit(X_mod_train, y_mod_train)
cm.score(X_mod_test, y_mod_test)
plt.title('Matriz de Confusão', fontsize=16);

print(classification_report(y_mod_test, previsoes))

#%% RANDOM FOREST

#SEM IPF
#rfc = RandomForestClassifier(n_estimators=60, criterion='gini', \
#    random_state = 1)

#COM IPF
rfc = RandomForestClassifier(n_estimators=60, criterion='gini', \
                             random_state = 1)
rfc.fit(X_mod_train, y_mod_train)

pickle.dump(rfc, open('../MODELOS/RFC/rfc_'+date_time+'.sav', 'wb'))

previsoes = rfc.predict(X_mod_test)

f3 = plt.figure(figsize=(19, 15))
cm = ConfusionMatrix(rfc)
cm.fit(X_mod_train, y_mod_train)
cm.score(X_mod_test, y_mod_test)
plt.title('Matriz de Confusão', fontsize=16);

print(classification_report(y_mod_test, previsoes))

#%% GRADIENT BOOSTING

#SEM IPF
#gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5,\
#    random_state = 1)

#COM IPF
gbc = GradientBoostingClassifier(n_estimators=10, learning_rate=1, \
                                 random_state = 1)
gbc.fit(X_mod_train, y_mod_train)

pickle.dump(gbc, open('../MODELOS/GBC/gbc_'+date_time+'.sav', 'wb'))

previsoes = gbc.predict(X_mod_test)

f3 = plt.figure(figsize=(19, 15))
cm = ConfusionMatrix(gbc)
cm.fit(X_mod_train, y_mod_train)
cm.score(X_mod_test, y_mod_test)
plt.title('Matriz de Confusão', fontsize=16);

print(classification_report(y_mod_test, previsoes))

#%% REDE NEURAL
         
#SEM IPF
#rna = MLPClassifier(verbose=True, early_stopping= True, max_iter=400, \
#    learning_rate_init = 0.001, tol = 0.0000010, hidden_layer_sizes = (32,), \
#        n_iter_no_change = 1000, random_state=1)

#COM IPF
rna = MLPClassifier(verbose=True, early_stopping= True, max_iter=1000,\
    learning_rate_init = 0.001, tol = 0.0000010, hidden_layer_sizes = (32,), \
        n_iter_no_change = 100, random_state=1)

rna.fit(X_mod_train, y_mod_train)

pickle.dump(rna, open('../MODELOS/RNA/rna_'+date_time+'.sav', 'wb'))

previsoes = rna.predict(X_mod_test)

f3 = plt.figure(figsize=(19, 15))
cm = ConfusionMatrix(rna)
cm.fit(X_mod_train, y_mod_train)
cm.score(X_mod_test, y_mod_test)
plt.title('Matriz de Confusão', fontsize=16);

print(classification_report(y_mod_test, previsoes))

f4 = plt.figure(figsize=(19, 15))
plt.plot(rna.loss_curve_)
plt.plot(rna.validation_scores_)

#%% OPTIMUM PATH FOREST

opf = SupervisedOPF(distance="log_squared_euclidean", \
                    pre_computed_distance=None)
opf.learn(X_mod_train, y_mod_train, X_mod_test ,y_mod_test, n_iterations=5) 

pickle.dump(opf, open('../MODELOS/OPF/opf_'+date_time+'.sav', 'wb'))

previsoes = opf.predict(X_mod_test)

f3 = plt.figure(figsize=(19, 15))
cm = metrics.confusion_matrix(y_mod_test, previsoes)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, \
                                            display_labels = [False, True])
cm_display.plot()
plt.title('Matriz de Confusão', fontsize=16);

print(classification_report(y_mod_test, previsoes))

#%% TESTES

with open(r"../MODELOS/SVM/modelo_20220706_185711.sav", "rb") as input_file:
    modelo = pickle.load(input_file)

previsoes = modelo.predict(X_mod_test)


f3 = plt.figure(figsize=(19, 15))
cm = ConfusionMatrix(svm)
cm.fit(X_mod_train, y_mod_train)
cm.score(X_mod_test, y_mod_test)
plt.title('Matriz de Confusão', fontsize=16);

print(classification_report(y_mod_test, previsoes))

#%% VALIDAÇÃO CRUZADA

#Autor das funções: Iniabasi Affiah
#Fonte: https://www.section.io/engineering-education/how-to-implement-k-fold-cross-validation/

def cross_validation(model, _X, _y, _cv=5):
    
      '''Function to perform 5 Folds Cross-Validation
       Parameters
       ----------
      model: Python Class, default=None
              This is the machine learning algorithm to be used for training.
      _X: array
           This is the matrix of features.
      _y: array
           This is the target variable.
      _cv: int, default=5
          Determines the number of folds for cross-validation.
       Returns
       -------
       The function returns a dictionary containing the metrics 'accuracy', \
           'precision',
       'recall', 'f1' for both training set and validation set.
      '''
      _scoring = ['accuracy', 'precision', 'recall', 'f1']
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
      return {"Training Accuracy scores": results['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean()
              }





def plot_result(x_label, y_label, plot_title, train_data, val_data):
        '''Function to plot a grouped bar chart showing the training and 
        validation results of the ML model in each fold after applying K-fold 
        cross-validation.
         Parameters
         ----------
         x_label: str, 
            Name of the algorithm used for training e.g 'Decision Tree'
          
         y_label: str, 
            Name of metric being visualized e.g 'Accuracy'
         plot_title: str, 
            This is the title of the plot e.g 'Accuracy Plot'
         
         train_result: list, array
            This is the list containing either training precision, accuracy,
            or f1 score.
        
         val_result: list, array
            This is the list containing either validation precision, accuracy,
            or f1 score.
         Returns
         -------
         The function returns a Grouped Barchart showing the training and 
         validation result in each fold.
        '''
        
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1ª Amostragem", "2ª Amostragem", "3ª Amostragem", \
                  "4ª Amostragem", "5ª Amostragem"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Treinamento')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Teste')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
        

resultcrossval = cross_validation(gbc, X_mod, y_mod, 5)

model_name = "Gradient Boosting"
plot_result(model_name,
            "Precisão",
            "",
            resultcrossval["Training Precision scores"],
            resultcrossval["Validation Precision scores"])

plot_result(model_name,
            "Sensibilidade",
            "",
            resultcrossval["Training Recall scores"],
            resultcrossval["Validation Recall scores"])

plot_result(model_name,
            "F1-Score",
            "",
            resultcrossval["Training F1 scores"],
            resultcrossval["Validation F1 scores"])

print("precisão = "+str(resultcrossval["Mean Validation Precision"]))
print("sensibilidade = "+str(+resultcrossval["Mean Validation Recall"]))
print("F1-score = "+str(resultcrossval["Mean Validation F1 Score"]))