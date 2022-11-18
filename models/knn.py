"""
    knn.py
    Autor: Iago Magalhães
    Descrição:
        - Classe KNN
        - Criação do modelo KNN
        - Treino do modelo KNN
        - Turning dos parâmetros
        - Cálculo das métricas de avaliação
"""
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class KNN:
    def __init__(self, base):
        """
            Método construtor da classe KNN
            :param base: base de dados
        """
        self.base = base
        self.regressKNN = None

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
        

    def gridKNN(self, modelo, X_train, y_train):
        """
            Método gridKNN, para turning dos parâmetros
            :param modelo:
            :param X_train:
            :param y_train:
        """
        k_list = list(range(1,11))
        parametros = dict(n_neighbors=k_list, metric= ['minkowski', 'euclidean', 'manhattan', 'hamming'])
        grid_knn = GridSearchCV(modelo, parametros, scoring='neg_root_mean_squared_error')
        grid_knn.fit(X_train, y_train)
        print("Melhor valor de K é {} com o valor de RMSE {} ".format(grid_knn.best_params_,grid_knn.best_score_))
        scores = grid_knn.cv_results_
        set_dados = scores.setdefault('mean_test_score')
        k = grid_knn.best_params_
        return k

    def model_knn(self, k):
        """
            Método knn, para criação do modelo KNN
            :param k:
        """
        regressor = KNeighborsRegressor(n_neighbors=k['n_neighbors'], metric=k['metric'])
        return regressor

    def treino(self):
        """
            Método treino, para treino do modelo KNN
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

        # --------------------------------- KNN ---------------------------------
        ini = time.time()
        k = self.gridKNN(KNeighborsRegressor(), X_train, y_train)
        fim = time.time()

        regressKNN = self.model_knn(k)

        iniTre = time.time()
        regressKNN.fit(X_train, y_train)
        fimTre = time.time()

        iniTes = time.time()
        resultadoKNN = regressKNN.predict(X_test)
        fimTes = time.time()

        self.regressKNN = regressKNN

        results = self.plotMetricas(y_test, resultadoKNN)
        print(f'Tempo para turning dos parâmetros {fim - ini} segundos')
        print(f'Tempo para treino do algoritmo {fimTre - iniTre} segundos')
        print(f'Tempo para teste do algoritmo {fimTes - iniTes} segundos')
        print('-----------------------------------------------')

        return results
    
    def params(self):
        """
            Método params, para retornar o modelo gerado
        """
        return self.regressKNN