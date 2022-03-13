from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
tqdm.pandas()
import scipy
from scipy.sparse import hstack
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
import pandas as pd
import numpy as np
import datetime
import pickle
from scipy import stats, optimize, interpolate
nltk.download('punkt')
from keras.models import load_model
import re
import string
from sklearn.preprocessing import StandardScaler
import csv
import pymongo
from pymongo import MongoClient
from flask import Flask, render_template, request, redirect, Response
from flask_cors import CORS, cross_origin
import json
from bson.json_util import dumps
import statsmodels.api as sm
import pickle
app = Flask(__name__)
def load_obj(name ):
    with open('./tools_2/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def stem(linea):
    ss = SnowballStemmer('spanish')
    words = word_tokenize(linea)
    root = []
    for w in words:
        root.append(ss.stem(w))
    return " ".join(root)


def fill_missing_values(df):
    """
    Reemplaza valores vacíos en cada columna

    Input: Dataframe con valores vacíos
    Output: Dataframe sin valores vacíos
    """
    # df['ADD1'].fillna('unk_add1', inplace=True)
    # df['ADD2'].fillna('unk_add2', inplace=True)
    # df['ADD3'].fillna('unk_add3', inplace=True)

    # df['ADD1'].replace('No_Informado','unk_add1',inplace=True)
    # df['ADD2'].replace('No_Informado','unk_add2',inplace=True)
    # df['ADD3'].replace('No_Informado','unk_add3',inplace=True)

    df['ID_MARCA'].fillna('unk_brand', inplace=True)
    df['ID_SUBCLASE'].fillna('unk_subclase', inplace=True)

    # df['DESC_PRODUC'].fillna('unk_desc', inplace=True)
    df['ID_MODELO'].fillna('unk_modelo', inplace=True)
    # df['ID_ESTILO'].fillna('unk_estilo', inplace=True)
    # df['COD_TEMPORADA'].fillna('unk_temporada', inplace=True)
    # df['NRUT_PROVE'].fillna('unk_nrut', inplace=True)

    # df['BPROPIA'].fillna('unk_bpropia', inplace=True)
    # df['CTIP_PRD'].fillna('unk_ctip', inplace=True)
    # df['TALLA'].fillna('unk_talla', inplace=True)
    # df['TALLA'].replace('No_Informado','unk_talla',inplace=True)
    # df['COLOR'].fillna('unk_color', inplace=True)
    # df['COLOR'].replace('No_Informado','unk_color',inplace=True)

    return df

def preprocess_descripcion(linea):
    """
    Description:
    This function will process the text data.
    This function will perform decontracting words, removing stop words, removing special characters and then apply stemming on the words in the sentence.

    Input: original sentence
    Output: processed sentence
    """
    linea = str(linea)
    linea = linea.replace('\\r', ' ')
    linea = linea.replace('\\n', ' ')
    linea = linea.replace('\\"', ' ')
    linea = re.sub('[^A-Za-z0-9]+', ' ', linea)
    # https://gist.github.com/sebleier/554280

    root_linea = stem(linea.lower().strip())
    return root_linea

def get_day_feature(df,campo):
    sparse_day = scipy.sparse.csr_matrix(df[campo].values)
    sparse_day = sparse_day.reshape(-1,1) # Now the shape will be (1111901, 1).
    print("Day:",sparse_day)
    return sparse_day

def get_len_feature(col_series, scaler_text_len=None):
    """
    Description:
    This funciton will calculate the word count of the text and standardize it.

    Input: Series, fitted scaler[optional; used during inference]
    Output: standardized text length for each product and object of the fitted scaler
    """
    text_len = col_series.apply(lambda x: len(x.split()))
    if scaler_text_len==None:
        scaler_text_len = StandardScaler()
        scaler_text_len.fit(text_len.values.reshape(-1, 1))
    text_len = scaler_text_len.transform(text_len.values.reshape(-1, 1))
    return text_len, scaler_text_len

def split_text(text):
    if text=='unk_subclase':
        return ["No Label", "No Label", "No Label","No Label"]
    return text[:3],text[3:5],text[5:7],text[7:]

def split_categories(df):
    """
    Desription:
    This function separates the categories into its three parts.
    Main category, Sub-category 1 and Sub-category 2
    Then it will remove the original category_name field.

    Input: Dataframe having category_name field
    Output: Dataframe with splitted categories
    """
    df['NIVEL_1'], df['NIVEL_2'], df['NIVEL_3'], df['NIVEL_4'] = zip(*df['ID_SUBCLASE'].apply(lambda x: split_text(x)))
    df = df.drop('ID_SUBCLASE', axis=1)
    return df

def vectorize_data(col_data, vectorizer=None):
    """
    Description:
    This funciton will vectorize the input column data.

    Input: dataframe column
    Output: one-hot encoded values and the fitted vectorizer
    """
    print("Col_data:",col_data)
    if vectorizer==None:
        vectorizer = TfidfVectorizer(ngram_range=(1,7), max_features=100000)
        vectorizer.fit(col_data)
    ohe_data = vectorizer.transform(col_data)
    print("OHE_DATA:", ohe_data)
    return ohe_data, vectorizer

def feature_pipeline(X_data, marca_vectorizer=None, modelo_vectorizer=None, desc_vectorizer=None, desc_scaler_len=None):
    """
    Description: This function will do all the feature engineering on the input X_data,
                and create a final data, ready for training.

    Input: Original input dataframe,
                    the fitted vectorizers for all categorical and text columns [optional: used during inference],
                    scalers [optional: used during inference]
    Output: Featurized data
    """
    print()
    print("pre-processing text data...")

    X_data['ID_MARCA'] = X_data['ID_MARCA'].progress_apply(lambda x: str(x).lower())
    X_data['ID_MODELO'] = X_data['ID_MODELO'].progress_apply(lambda x: str(x).lower())
    X_data['DESC_ESTILO'] = X_data['DESC_ESTILO'].progress_apply(lambda x: str(x).lower())


    print('Getting word lengths')
    #texto_len, texto_scaler_len = get_len_feature(X_data['ID_MARCA'], texto_scaler_len)
    #texto_len, texto_scaler_len = get_len_feature(X_data['ID_MODELO'], texto_scaler_len)
    _, desc_scaler_len = get_len_feature(X_data['DESC_ESTILO'], desc_scaler_len)

    #print("Getting sparse day data...")
    #sparse_monday = get_day_feature(X_data,"C_MONDAY")
    #sparse_c_day = get_day_feature(X_data,"C_DAY")

    print("OHE vectorizing texto")
    marca_ohe, marca_vectorizer = vectorize_data(X_data['ID_MARCA'].values.astype('U'), marca_vectorizer)
    modelo_ohe, modelo_vectorizer = vectorize_data(X_data['ID_MODELO'].values.astype('U'), modelo_vectorizer)
    desc_ohe, desc_vectorizer = vectorize_data(X_data['DESC_ESTILO'].values.astype('U'), desc_vectorizer)
    print("Texto done...")

    print("Creating the final featurized dataset...")
    X_featurized = hstack((marca_ohe,
                           modelo_ohe,
                           desc_ohe,
                           X_data['DIA'].values.reshape(-1,1),
                           X_data['MES'].values.reshape(-1,1),
                           X_data['DESCUENTO'].values.reshape(-1,1)
                           #sparse_monday,
                           #sparse_c_day
                          )).tocsr()

    print("Done!!!\n---------------------------\n")
    print(X_featurized.shape)
    return X_featurized, marca_vectorizer, modelo_vectorizer, desc_vectorizer, desc_scaler_len


def feature_pipeline2(X_data, texto_vectorizer=None, texto_scaler_len=None, desc_vectorizer=None, desc_scaler_len=None):
    """
    Description: This function will do all the feature engineering on the input X_data,
                and create a final data, ready for training.

    Input: Original input dataframe,
                    the fitted vectorizers for all categorical and text columns [optional: used during inference],
                    scalers [optional: used during inference]
    Output: Featurized data
    """
    print()
    print("pre-processing text data...")

    X_data['Descripcion'] = X_data['Descripcion'].progress_apply(lambda x: str(x).lower())
    X_data['Texto'] = X_data['Texto'].progress_apply(lambda x: str(x).lower())

    print('Getting word lengths')
    texto_len, texto_scaler_len = get_len_feature(X_data['Texto'], texto_scaler_len)
    desc_len, desc_scaler_len = get_len_feature(X_data['Descripcion'], desc_scaler_len)
    # print("Getting sparse day data...")
    # sparse_monday = get_day_feature(X_data,"C_MONDAY")
    # sparse_c_day = get_day_feature(X_data,"C_DAY")

    print("OHE vectorizing texto")
    texto_ohe, texto_vectorizer = vectorize_data(X_data['Texto'].values.astype('U'), texto_vectorizer)
    desc_ohe, desc_vectorizer = vectorize_data(X_data['Descripcion'].values.astype('U'), desc_vectorizer)
    print("Texto done...")

    print("Creating the final featurized dataset...")
    X_featurized = hstack((texto_ohe,
                           desc_ohe
                           # sparse_monday,
                           # sparse_c_day
                           )).tocsr()

    print("Done!!!\n---------------------------\n")
    print(X_featurized.shape)
    return X_featurized, texto_vectorizer, texto_scaler_len, desc_vectorizer, desc_scaler_len

def get_database():

    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    CONNECTION_STRING = "mongodb://127.0.0.1:27017/"

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    client = MongoClient(CONNECTION_STRING)

    # Create the database for our example (we will use the same database throughout the tutorial
    return client.price2be

ruta_modelo_inicio = "./Modelos/Entrega/"
ruta_vectorizadores = "./Vectorizadores/"
rutas = ['J11',
         'J04',
         'J03',
         'J01',
         'J08',
         'J09',
         'J12',
         'J10',
         'J02',
         'J15',
         'J05',
         'J06',
         'J17',
         'J07',
         'J16',
         'J99_J14_J18_J95_J98_J13_J32'
         ]

lista_modelos = {
    'J11': 0,
    'J04': 1,
    'J03': 2,
    'J01': 3,
    'J08': 4,
    'J09': 5,
    'J12': 6,
    'J10': 7,
    'J02': 8,
    'J15': 9,
    'J05': 10,
    'J06': 11,
    'J17': 3,
    'J07': 3,
    'J16': 5,
    'J99': 15,
    'J14': 15,
    'J18': 15,
    'J95': 15,
    'J98': 15,
    'J13': 15,
    'J32': 15
}

caso = {
    'ID': 1404182,
    'ID_SUBCLASE': 'J11140201',
    'ID_MARCA': 'targus',
    'ID_MODELO': 'tsb226us',
    'DESC_ESTILO': 'Mochila Terra Targus',
    'DIA': 14,
    'MES': 9,
    'C_MONDAY': 0,
    'C_DAY': 0,
    'NIVEL_1': 'J11',
    'NIVEL_2': '14',
    'NIVEL_3': '02',
    'NIVEL_4': '01'}

caso_de_prueba = pd.DataFrame(caso, index=[0])
print(caso_de_prueba)


nuevo = load_obj("cantidades_por_categoria")
print(nuevo['J11050102'])

dic = dict()
dic["Mochilas y Bolsos"] = "J11140201"
dic["Utensilios de Cocina"] = "J11090124"
dic["Electronica"] = "J11141201"
dic["Electrodomesticos, Para Cocina"] = "J11090124"
dic["Moda"] = "J05020102"


CORS(app)

@app.route('/receiver', methods=['POST'])
def worker():
    # read json + reply
    data = request.get_json()
    descripcion = data['descripcion']
    nombre = data['nombre'].lower()
    marca = data['marca'].lower()
    modelo = data['modelo'].lower()
    dia = int(data['dia'])
    mes = int(data['mes'])
    categoria = dic[data['categoria']]
    # n1, n2, n3, n4 = split_text(categoria)

    # nombre = traducir(nombre)
    input_modelo = {
        'ID_SUBCLASE': categoria,
        'ID_MARCA': marca,
        'ID_MODELO': modelo,
        'DESC_ESTILO': nombre,
        'DIA': dia,
        'MES': mes,
        'C_MONDAY': 0,
        'C_DAY': 0,
        'DESCUENTO': 0
        # 'NIVEL_1': 'J11',
        # 'NIVEL_2': '14',
        # 'NIVEL_3': '02',
        # 'NIVEL_4': '01'
    }

    cat = input_modelo['ID_SUBCLASE'][0:3]
    nombre_modelo = rutas[lista_modelos[cat]]
    ruta = ruta_modelo_inicio + nombre_modelo
    modelo = load_model(ruta)

    marca_vectorizer = pickle.load(open(ruta_vectorizadores + cat + "/marca_vectorizer.pickle", "rb"))
    modelo_vectorizer = pickle.load(open(ruta_vectorizadores + cat + "/modelo_vectorizer.pickle", "rb"))
    desc_vectorizer = pickle.load(open(ruta_vectorizadores + cat + "/desc_vectorizer.pickle", "rb"))
    desc_scaler_len = pickle.load(open(ruta_vectorizadores + cat + "/desc_scaler_len.pickle", "rb"))

    caso_de_prueba = pd.DataFrame(input_modelo, index=[0])

    x_probando, _, _, _, _ = feature_pipeline(caso_de_prueba, marca_vectorizer=marca_vectorizer,
                                              modelo_vectorizer=modelo_vectorizer, desc_vectorizer=desc_vectorizer,
                                              desc_scaler_len=desc_scaler_len)

    xp = modelo.predict(x_probando)[0][0]

    print(type(xp))
    resultado1 = np.exp(xp)

    print("Modelo viejo:" + str(resultado1))

    print(descripcion)

    result = str(round(resultado1)) + ";" + str(nuevo[categoria])
    # print(type(data))

    '''
    if data != None:
        #data['category_name'].append()
        if data['categoria'] == 'Videojuegos':

            # loop over every row
            result = str(np.exp(new_model.predict(test[54047])))
    '''
    return result


@app.route('/marcas', methods=['POST'])
def worker2():
    db = get_database()
    data = request.get_json()
    nombre = data['nombre']
    pipeline = [
        {"$match": {"$text": {"$search": nombre}}},
        {"$sort": {"score": {"$meta": "textScore"}}},
        {"$project": {"nombre": 1, "marca": 1, "_id": 0, "score": {"$meta": "textScore"}}},
        {"$match": {"score": {"$gte": 0.6}}},
        {"$group": {"_id": "$marca", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    lista = list(db.catalogo_productos.aggregate(pipeline, allowDiskUse=True))
    if len(lista) > 10:
        lista2 = list(lista[:9])
        lista3 = list(lista[9:])
        c = 0
        for e in lista3:
            c += e["count"]
        lista2.append({'_id': 'Otros', 'count': c})
        lista = lista2
    res_marcas = dict()
    for r in lista:
        res_marcas[r['_id']] = r['count']
    print(res_marcas)
    res = json.dumps(res_marcas)
    return res


@app.route('/cantidad', methods=['POST'])
def worker3():
    db = get_database()
    data = request.get_json()
    nombre = data['nombre']
    pipeline = [
        {"$match": {"$text": {"$search": nombre}}},
        {"$sort": {"score": {"$meta": "textScore"}}},
        {"$project": {"nombre": 1, "_id": 0, "score": {"$meta": "textScore"}}},
        {"$match": {"score": {"$gte": 2.0}}},
        {"$group": {"_id": "null", "count": {"$sum": 1}}}
    ]
    results = db.catalogo_productos.aggregate(pipeline, allowDiskUse=True)
    res = dict()
    for row in results:
        res['count'] = row['count']

    pipeline = [
        {"$match": {"$text": {"$search": nombre}}},
        {"$sort": {"score": {"$meta": "textScore"}}},
        {"$project": {"nombre": 1, "dominio": 1, "link": 1, "_id": 0, "score": {"$meta": "textScore"}}},
        {"$match": {"score": {"$gte": 2.0}}},
        {"$limit": 3}
    ]
    results = db.catalogo_productos.aggregate(pipeline, allowDiskUse=True)
    c = 1
    for row in results:
        res['link' + str(c)] = row['dominio'] + ";" + row['nombre'] + ";" + row['link']
        c += 1
    return res


@app.route('/precios', methods=['POST'])
def worker6():
    db = get_database()
    data = request.get_json()
    nombre = data['nombre']
    pipeline = [
        {"$match": {"$text": {"$search": nombre}}},
        {"$sort": {"score": {"$meta": "textScore"}}},
        {"$project": {"_id": 0, "precio": 1, "score": {"$meta": "textScore"}}},
        {"$match": {"score": {"$gte": 2.5}}},
        {"$project": {"_id": 0, "precio": 1}}
    ]

    lista = list(db.catalogo_productos.aggregate(pipeline, allowDiskUse=True))

    if (len(lista) == 0):
        return

    else:
        lista2 = list()
        for r in lista:
            lista2.append(r['precio'].replace("$", "").replace(".", ""))

        res = json.dumps(lista2)

    return res


@app.route('/monthly', methods=['POST'])
def worker4():
    # read json + reply
    responseData = []
    data = request.get_json()
    descripcion = data['descripcion']
    nombre = data['nombre'].lower()
    marca = data['marca'].lower()
    modelo = data['modelo'].lower()
    dia = int(data['dia'])
    mes = int(data['mes'])
    categoria = dic[data['categoria']]

    annio = datetime.datetime.now().year
    fecha = datetime.datetime(annio, mes, dia)
    for i in range(0, 17):
        fecha = fecha + datetime.timedelta(days=7)
        dia = fecha.day
        mes = fecha.month
        stringFecha = fecha.strftime("%x")

        input_modelo = {
            'ID_SUBCLASE': categoria,
            'ID_MARCA': marca,
            'ID_MODELO': modelo,
            'DESC_ESTILO': nombre,
            'DIA': dia,
            'MES': mes,
            'C_MONDAY': 0,
            'C_DAY': 0,
            'DESCUENTO': 0
        }

        cat = input_modelo['ID_SUBCLASE'][0:3]
        nombre_modelo = rutas[lista_modelos[cat]]
        ruta = ruta_modelo_inicio + nombre_modelo
        modelo = load_model(ruta)

        marca_vectorizer = pickle.load(open(ruta_vectorizadores + cat + "/marca_vectorizer.pickle", "rb"))
        modelo_vectorizer = pickle.load(open(ruta_vectorizadores + cat + "/modelo_vectorizer.pickle", "rb"))
        desc_vectorizer = pickle.load(open(ruta_vectorizadores + cat + "/desc_vectorizer.pickle", "rb"))
        desc_scaler_len = pickle.load(open(ruta_vectorizadores + cat + "/desc_scaler_len.pickle", "rb"))

        caso_de_prueba = pd.DataFrame(input_modelo, index=[0])

        x_probando, _, _, _, _ = feature_pipeline(caso_de_prueba, marca_vectorizer=marca_vectorizer,
                                                  modelo_vectorizer=modelo_vectorizer, desc_vectorizer=desc_vectorizer,
                                                  desc_scaler_len=desc_scaler_len)

        xp = modelo.predict(x_probando)[0][0]

        resultado1 = np.exp(xp)

        result = str(round(resultado1))
        responseData.append((stringFecha, result))

    return json.dumps(responseData)


@app.route('/estimacion', methods=['POST'])
def worker_est():
    db = get_database()
    data = request.get_json()
    score = 1.5
    pipeline = [
        {"$match": {"$text": {"$search": data['nombre']}}},
        {"$sort": {"score": {"$meta": "textScore"}}},
        {"$project": {"nombre": 1, "_id": 0, "01-17": 1, "02-17": 1, "03-17": 1, "04-17": 1, "05-17": 1,
                      "06-17": 1, "07-17": 1, "08-17": 1, "09-17": 1, "10-17": 1, "11-17": 1, "12-17": 1,
                      "01-18": 1, "02-18": 1, "03-18": 1, "04-18": 1, "05-18": 1,
                      "06-18": 1, "07-18": 1, "08-18": 1, "09-18": 1, "10-18": 1, "11-18": 1, "12-18": 1,
                      "score": {"$meta": "textScore"}}},
        {"$match": {"score": {"$gte": score}}},
        {"$limit": 1}
    ]
    results = db.catalogo_historico.aggregate(pipeline, allowDiskUse=True)
    c = 1
    ventas = list()
    for row in results:
        print(row)
        for a in range(1, 10):
            if '0' + str(a) + '-17' in row.keys():
                ventas.append(int(row['0' + str(a) + '-17']))
            else:
                ventas.append(0)
        if '10-17' in row.keys():
            ventas.append(int(row['10-17']))
        else:
            ventas.append(0)
        if '11-17' in row.keys():
            ventas.append(int(row['11-17']))
        else:
            ventas.append(0)
        if '12-17' in row.keys():
            ventas.append(int(row['12-17']))
        else:
            ventas.append(0)
        for a in range(1, 10):
            if '0' + str(a) + '-18' in row.keys():
                ventas.append(int(row['0' + str(a) + '-18']))
            else:
                ventas.append(0)
        if '10-18' in row.keys():
            ventas.append(int(row['10-18']))
        else:
            ventas.append(0)
        if '11-18' in row.keys():
            ventas.append(int(row['11-18']))
        else:
            ventas.append(0)
        if '12-18' in row.keys():
            ventas.append(int(row['12-18']))
        else:
            ventas.append(0)
        break
    print(ventas)
    if len(ventas) == 0:
        ventas = np.zeros(24)
    resp = dict()
    resp['meses'] = ['Diciembre', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio']
    resp['meses_ful'] = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre',
                         'Octubre', 'Noviembre', 'Diciembre']
    avg = list()
    for i in range(12):
        avg.append(np.floor(ventas[i] * 0.33 + ventas[i + 12] * 0.67))
    # print(avg)
    resp['avg'] = avg
    resp['ventas'] = ventas[12:]
    x = range(1, 13)
    linre = scipy.stats.linregress(x, avg)
    res = list()
    for u in range(13, 20):
        res.append(linre.intercept + linre.slope * u)
    r_squared = linre.rvalue ** 2
    resp['linre'] = res
    resp['r_squared'] = round(r_squared, 3)
    # print('r_squared: '+str(r_squared))
    # print(res)

    x_ful = range(1, 25)
    linre_ful = scipy.stats.linregress(x_ful, ventas)
    res_ful = list()
    for u in range(25, 32):
        val = linre_ful.intercept + linre_ful.slope * u
        if val < 0:
            res_ful.append(0)
        else:
            res_ful.append(linre_ful.intercept + linre_ful.slope * u)
    r_squared_ful = linre.rvalue ** 2
    # print(res_ful)

    resp = json.dumps(resp)
    return resp


@app.route('/sarimax', methods=['POST'])
def worker_sar():
    db = get_database()
    data = request.get_json()
    score = 1.5
    pipeline = [
        {"$match": {"$text": {"$search": data['nombre']}}},
        {"$sort": {"score": {"$meta": "textScore"}}},
        {"$project": {"nombre": 1, "_id": 0, "01-17": 1, "02-17": 1, "03-17": 1, "04-17": 1, "05-17": 1,
                      "06-17": 1, "07-17": 1, "08-17": 1, "09-17": 1, "10-17": 1, "11-17": 1, "12-17": 1,
                      "01-18": 1, "02-18": 1, "03-18": 1, "04-18": 1, "05-18": 1,
                      "06-18": 1, "07-18": 1, "08-18": 1, "09-18": 1, "10-18": 1, "11-18": 1, "12-18": 1,
                      "score": {"$meta": "textScore"}}},
        {"$match": {"score": {"$gte": score}}},
        {"$limit": 1}
    ]
    results = db.catalogo_historico.aggregate(pipeline, allowDiskUse=True)
    c = 1
    ventas = list()
    for row in results:
        print(row)
        for a in range(1, 10):
            if '0' + str(a) + '-17' in row.keys():
                ventas.append(int(row['0' + str(a) + '-17']))
            else:
                ventas.append(0)
        if '10-17' in row.keys():
            ventas.append(int(row['10-17']))
        else:
            ventas.append(0)
        if '11-17' in row.keys():
            ventas.append(int(row['11-17']))
        else:
            ventas.append(0)
        if '12-17' in row.keys():
            ventas.append(int(row['12-17']))
        else:
            ventas.append(0)
        for a in range(1, 10):
            if '0' + str(a) + '-18' in row.keys():
                ventas.append(int(row['0' + str(a) + '-18']))
            else:
                ventas.append(0)
        if '10-18' in row.keys():
            ventas.append(int(row['10-18']))
        else:
            ventas.append(0)
        if '11-18' in row.keys():
            ventas.append(int(row['11-18']))
        else:
            ventas.append(0)
        if '12-18' in row.keys():
            ventas.append(int(row['12-18']))
        else:
            ventas.append(0)
        break
    print(ventas)
    if len(ventas) == 0:
        ventas = np.zeros(22)
    resp = dict()
    resp['meses'] = ['Marzo', 'Abril', 'Mayo', 'Junio','Julio', 'Agosto', 'Septiembre']
    model = sm.tsa.statespace.SARIMAX(ventas, order=(5, 1, 0), seasonal_order=(1, 1, 2, 12))
    results = model.fit()
    ress=results.predict(start=1,end=31,dynamic=False)
    temp = list(ress[15:22])
    # temp = np.where(temp<0, 0, temp)
    for i in range(len(temp)):
        if temp[i] < 0:
            temp[i] = 0
    for i in range(len(ress)):
        if ress[i] < 0:
            ress[i] = 0
    resp['SAR'] = list(temp)
    mse=mean_squared_error(ventas[:22],ress[:22])
    msle=mean_squared_log_error(ventas[:22],ress[:22])
    resp['msle']=msle
    resp['mse']=mse
    resp = json.dumps(resp)
    return resp


@app.route('/tags', methods=['POST'])
def worker_tags():
    db = get_database()
    data = request.get_json()
    score = 2.0

    pipeline = [
        {"$match": {"$text": {"$search": data['nombre']}}},
        {"$sort": {"score": {"$meta": "textScore"}}},
        {"$project": {"nombre": 1, "marca": 1, "tags": 1, "_id": 0, "score": {"$meta": "textScore"}}},
        {"$match": {"score": {"$gte": score}}}
    ]
    lista = list(db.catalogo_productos.aggregate(pipeline, allowDiskUse=True))
    counter = dict()
    for item in lista:
        tags = item['tags']
        if '/' in tags:
            tags = tags.split('/')
        else:
            tags = tags.split(',')
        for tag in tags:
            tag = tag.strip()
            if tag == '':
                continue
            if tag not in counter.keys():
                counter[tag] = 0
            counter[tag] += 1

    res = list()
    for t in sorted(list(counter.items()), key=lambda x: x[1], reverse=True)[:5]:
        res.append(' ' + t[0] + ',')
    res[-1] = res[-1].strip(',')
    res = json.dumps(res[:5])
    return res


@app.route('/history', methods=['POST'])
def worker_4():
    db = get_database()
    data = request.get_json()
    db.publicaciones.insert_one(data)
    return 0


@app.route('/archived', methods=['POST'])
def worker5():
    db = get_database()
    data = request.get_json()
    mail = data["email"]
    print(mail)
    cursor = db.publicaciones.find({"$and": [{'activa': 1}, {'email': data['email']}]})
    list_cur = list(cursor)
    json_data = dumps(list_cur)
    print(json_data)
    return json_data


@app.route('/pricechange', methods=['POST'])
def worker_6():
    db = get_database()
    data = request.get_json()
    filtro = {"$and": [{"nombre": data["nombre"]}, {'email': data['email']}, {'activa': 1}]}
    print(filtro)
    nuevo = {"$set": {"precio": data["precio"]}}
    print(nuevo)
    db.publicaciones.update_one(filtro, nuevo)
    print("worker6")
    return dumps(data)


@app.route('/activechange', methods=['POST'])
def worker7():
    db = get_database()
    data = request.get_json()
    filtro = {"$and": [{"nombre": data["nombre"]}, {'email': data['email']}, {'activa': 1}]}
    print(filtro)
    nuevo = {"$set": {"activa": 0}}
    print(nuevo)
    db.publicaciones.update_one(filtro, nuevo)
    print("worker7")
    return dumps(data)


@app.route('/terminadas', methods=['POST'])
def worker8():
    db = get_database()
    data = request.get_json()
    cursor = db.publicaciones.find({"$and": [{'activa': 0}, {'email': data['email']}]})
    list_cur = list(cursor)
    json_data = dumps(list_cur)
    print(json_data)
    return json_data


@app.route('/sugerencias', methods=['POST'])
def worker9():
    db = get_database()
    data = request.get_json()
    filtro = {"$and": [{"nombre": data["nombre"]}, {'email': data['email']}]}
    nuevo = {"$set": {"sugerencias": data["sugerencias"]}}
    db.publicaciones.update_one(filtro, nuevo)
    nuevo = {"$set": {"fechas_sug": data["fechas_sug"]}}
    db.publicaciones.update_one(filtro, nuevo)

    return dumps(data)


if __name__ == "__main__":
    app.run()
