"""
    tdr.py
    Autor: Iago Magalhães
    Descrição:
        - Classe TDR
        - Criação do modelo TDR
        - Treino do modelo TDR
        - Turning dos parâmetros
        - Cálculo das métricas de avaliação
"""
import time

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class TDR:
    def __init__(self, base):
        """
            Método construtor da classe TDR
            :param base: base de dados
        """
        self.base = base
        self.regressTDR = None

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

    def model_tdr(self):
        """
            Método knn, para criação do modelo TDR
            :param k:
        """
        regressor = DecisionTreeRegressor()
        return regressor

    def treino(self):
        """
            Método treino, para treino do modelo TDR
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

        # --------------------------------- TDR ---------------------------------
        regressTDR = self.model_tdr()

        iniTre = time.time()
        regressTDR.fit(X_train, y_train)
        fimTre = time.time()

        iniTes = time.time()
        resultadoTDR = regressTDR.predict(X_test)
        fimTes = time.time()

        self.regressTDR = regressTDR

        results = self.plotMetricas(y_test, resultadoTDR)
        print(f'Tempo para treino do algoritmo {fimTre - iniTre} segundos')
        print(f'Tempo para teste do algoritmo {fimTes - iniTes} segundos')
        print('-----------------------------------------------')

        return results

    def params(self):
        """
            Método params, para retornar o modelo gerado
        """
        return self.regressTDR