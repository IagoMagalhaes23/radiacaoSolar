"""
    elmr.py
    Autor: Iago Magalhães
    Descrição:
        - Classe ELMR
        - Criação do modelo ELMR
        - Treino do modelo ELMR
        - Turning dos parâmetros
        - Cálculo das métricas de avaliação
"""
import time
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from skelm import ELMRegressor

class ELMR:
    def __init__(self, base):
        """
            Método construtor da classe ELMR
            :param base: base de dados
        """
        self.base = base
        self.regressELMR = None

    def plotMetricas(self, y_test, resultado):
        """
            Método plotMetricas, para plotagem da métricas de avaliação
            :param y_test:
            :param resultado:
        """
        print('----------------------------------------')
        print('Métricas de avaliação')
        MSE = mean_squared_error(y_test, resultado)
        print(f"MSE: {MSE}")
        RMSE = mean_squared_error(y_test, resultado, squared=False)
        print(f"RMSE: {RMSE}")
        MAE = mean_absolute_error(y_test, resultado)
        print(f"MAE: {MAE}")
        R2SCORE = r2_score(y_test, resultado)
        print(f"Valor do R2SCORE: {R2SCORE}")
        print('----------------------------------------')

        results = [R2SCORE]
        return results
        

    def gridELMR(self, X_train, y_train, X_test, y_test):
        """
            Método gridELMR, para turning dos parâmetros
            :param modelo:
            :param X_train:
            :param y_train:
        """
        neur_possiveis = np.arange(1,1000,1)
        MSE_total = []

        for neuronios in neur_possiveis:
            elmr = ELMRegressor(n_neurons = neuronios, ufunc='tanh')    
            elmr.fit(X_train, y_train)
            predicao = elmr.predict(X_test)

            MSE = mean_squared_error(y_test, predicao)

            MSE_total.append(MSE)

        qtd_neuronios = neur_possiveis[MSE_total.index(min(MSE_total))]
        print(qtd_neuronios)
        return qtd_neuronios

    def model_elmr(self, neuronios):
        """
            Método knn, para criação do modelo ELMR
            :param k:
        """
        regressor = ELMRegressor(n_neurons=neuronios, ufunc='tanh')
        return regressor

    def treino(self):
        """
            Método treino, para treino do modelo ELMR
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.base['Radiacao (KJ/mï¿½)'], self.base['Radiacao (KJ/mï¿½)'],
            test_size=0.46, shuffle=False
            )
        #15 horas correspondem a 46% dos dados

        X_train= X_train.iloc[:].values
        X_test = X_test.iloc[:].values
        X_train = X_train.reshape(-1,1)
        X_test = X_test.reshape(-1,1)

        # --------------------------------- ELMR ---------------------------------
        ini = time.time()
        qtd_neuronios = self.gridELMR(X_train, y_train, X_test, y_test)
        fim = time.time()

        regressELMR = self.model_elmr(qtd_neuronios)

        iniTre = time.time()
        regressELMR.fit(X_train, y_train)
        fimTre = time.time()

        iniTes = time.time()
        resultadoELMR = regressELMR.predict(X_test)
        fimTes = time.time()

        self.regressELMR = regressELMR

        results = self.plotMetricas(y_test, resultadoELMR)
        print(f'Tempo para turning dos parâmetros {fim - ini} segundos')
        print(f'Tempo para treino do algoritmo {fimTre - iniTre} segundos')
        print(f'Tempo para teste do algoritmo {fimTes - iniTes} segundos')
        print('-----------------------------------------------')

        return results
    
    def params(self):
        """
            Método params, para retornar o modelo gerado
        """
        return self.regressELMR