"""
    @Autor: Iago Magalhães
    @Descrição:
        - Seleção de melhor modelo de predição
        - Treinamento de algoritmos
        - Realizar predições
"""
from flask import Flask, request
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from models.elmr import ELMR
from models.knn import KNN
from models.tdr import TDR

app = Flask(__name__)

def melhorModelo(metrics):
    """
        Função para selecionar o modelo com melhores resultados de R2SCORE
        :param metrics: list com os valores das métricas
    """
    max_value = max(metrics)
    
    return metrics.index(max_value)

@app.route('/treinar')
def treinar():
    """
        Função para realizar treinamento do modelo
        :param dados: dados utilizados para treinamento
    """
    dados = "data.csv"
    data = pd.read_csv(dados)

    mdl1 = KNN(data)
    mdl2 = TDR(data)
    mdl3 = ELMR(data)
    
    models = [mdl1.treino(), mdl2.treino(), mdl3.treino()]

    pos = melhorModelo(models)
    print(pos)
    if(pos == 0):
        regres = mdl1.params()
        joblib.dump(regres, 'modelo.pkl')
        return 'Modelo KNN gerado'
    elif(pos == 1):
        regres = mdl2.params()
        joblib.dump(regres, 'modelo.pkl')
        return 'Modelo TDR gerado'
    elif(pos == 2):
        regres = mdl2.params()
        joblib.dump(regres, 'modelo.pkl')
        return 'Modelo ELMR gerado'

@app.route('/predicoes')
def predicoes():
    """
        Função para realizar predições com o modelo de Machine Learning
    """
    model = 'modelo.pkl'
    dado = request.args.get('dado')
    print(f'Modelo selecionado: {model}')
    print(f'Dado recebido: {dado}')

    data = np.array(dado)

    data = data.reshape(1,-1)

    modelPred = joblib.load(model)
    previsao = modelPred.predict(data)
    print(previsao[0])
    return f'Previsão: {str(previsao[0])}'

@app.route('/')
def index():
    """
        Função para a seleção de momento de treinamento e seleção dos algoritmos
    """
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%H:%M:%S")

    if((currentTime[0:5] == '03:00') and (datetime.today().weekday() == 6)): #Se for verdade, se for data e horário, realiza o treinamento dos modelos
        treinar()
        return 'Sistema indisponível no momento'
    else: #Se não for verdade, chama a função 'predicoes' para realizar as previsões
        return 'Acesse o link para realizar a predição do valor: 127.0.0.1:5000/predicoes'

if __name__=="__main__":
    app.run(debug=True)