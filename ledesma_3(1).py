import kagglehub
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix,precision_score,  recall_score,  f1_score
import matplotlib.pyplot as plt
import seaborn as sns
#Cargamos el dataset desde kaggle y obtenemos el path del archivo.
path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
print(path)
#En este punto convertimos el dataset en un dataframe y revisamos las características del mismo.
df = pd.read_csv("/kaggle/input/-spotify-tracks-dataset/dataset.csv")
print(df.info())

df=df[df["popularity"]!=0]
#Eliminamos las filas que tuvieran como popularidad 0 para evitar reproducir canciones poco conocidas en la radio.
print(df.head(10))
print(df.info())

def agrupar_porgenero(genero):
    if genero in ["pop", "dance", "disco", "power-pop", "cantopop", "indie-pop", "k-pop", "power-pop"]:
        return "pop_comercial"
    elif genero in ["rock", "alt-rock", "punk", "hard-rock", "metal", "bluegrass", "j-rock", "punk-rock", "rock-n-roll"]:
        return "rock"
    elif genero in ["latin", "salsa", "reggaeton", "latino", "brazil", "reggae", "tango"]:
        return "latino"
    elif genero in ["edm", "electro", "electronic", "techno", "house", "trance", "afrobeat", "idm", "breakbeat", "club", "deephouse"]:
        return "electronica"
    elif genero in ["acoustic", "ambient", "piano", "classical", "sleep","chill", "jazz"]:
        return "tranquila"
    elif genero in ["hip-hop", "r-n-b", "rap", "dancehall"]:
        return "urbana"
    else:
        return "otro"
#Esta función de agrupación de géneros la creamos con el propósito de juntar los subgéneros dentro de un género paraguas.
df["Género"] = df["track_genre"].apply(agrupar_porgenero)
#Creamos una nueva columna llamada categoría, la cual contiene el género paraguas de cada canción.
print("Cantidad de canciones por género:")
print("\n")
print(df["Género"].value_counts())

df = df[df["Género"] != "otro"]
#Eliminamos todos las filas que tuvieran como género paraguas otro, ya que es demasiada mezcla entre géneros no importantes para la estación de radio del abuelo.
#También consideramos importante eliminar estos datos porque pueden presentar demasiado ruido para nuestro modelo.
print(df.shape)
print("Clasificación final de géneros:")
print("\n")
print(df["Género"].value_counts())

x=df[["popularity","duration_ms","danceability","energy","loudness","speechiness","acousticness","instrumentalness","liveness","valence","tempo"]]
y=df["Género"]
#Aquí seleccionamos nuestras variables predictoras y nuesta variable objetivo. Seleccionamos 11 variables para predecir el género.
x_entren, x_prueba, y_entren, y_prueba = train_test_split(x,y,test_size=0.2,random_state=42)
#Con más de 34000 datos consideramos que una separación de 80% - 20% de los datos para entrenamiento y prueba era suficiente para el modelo.
print("Datos de entrenamiento: ", x_entren.shape)
print("Datos de prueba:", x_prueba.shape)

Naive_Bayes = GaussianNB()
Naive_Bayes.fit(x_entren, y_entren)
#Primero entrenamos un modelo basado en el teorema de Naive Bayes, el cual supone que todos las variables son independientes entre sí.
predicción_Naive = Naive_Bayes.predict(x_prueba)
precisión_Naive = accuracy_score(y_prueba, predicción_Naive)
print("Resultado de precisión de Naive-Bayes")
print(precisión_Naive)
print("\n")
print("Reporte de modelo Naive-Bayes:")
print(classification_report(y_prueba, predicción_Naive))

modelo_logistica = LogisticRegression()
modelo_logistica.fit(x_entren, y_entren)
#Aquí entrenamos un modelo de regresión logística, el cual mide la probabilidad de pertenecer a cada género paraguas.
#Debido a la cantidad de datos decidimos utilizar max_iter=1000 para asegurar que haya convergencia.
predicción_logistica = modelo_logistico.predict(x_prueba)
precisión_logistica = accuracy_score(y_prueba, predicción_logistica)
print("Resultado de precisión del modelo de Regresión Logística:")
print(precisión_logistica)
print("\n")
print("Reporte de modelo Regresión Logística:")
print(classification_report(y_prueba, predicción_logistica))
print("\n")

RandomForest = RandomForestClassifier(n_estimators=100,random_state=1,max_depth=20)
#Finalmente, entrenamos un modelo de random forest con 100 árboles de decisión, una profundidad máxima de cada árbol de 20 nodos,
#ya que es un valor intermedio y evita el overfitting. El random state simplemente lo fijamos a 1 para asegurar que los resultados no cambien.
RandomForest.fit(x_entren, y_entren)
predicción_RandomForest = RandomForest.predict(x_prueba)
precisión_RandomForest = accuracy_score(y_prueba, predicción_RandomForest)
print("Precisión del Random Forest:")
print(precisión_RandomForest)
print("\n")
print("Reporte del modelo Random Forest:")
print(classification_report(y_prueba, predicción_RandomForest))
print("\n")
print("Comparación de modelos:")
print("Naive Bayes:", precisión_Naive)
print("Regresión Logística:", precisión_logistica)
print("Random Forest:", precisión_RandomForest)
#Aquí finalmente comparamos la precisión de los 3 modelos para elegir cuál es el mejor.
#Observando la precisión de los 3 modelos, creamos una variable llamada mejorModelo 
#y una serie de ifs para decidir el modelo para nuestras predicciones.
mejorModelo = modelo_RandomForest
mejorNombre = "Random Forest"
if precisión_Naive > precisión_RandomForest and precisión_Naive > precisión_logistica:
    mejorModelo = modelo_Naive
    mejorNombre = "Naive Bayes"
if precisión_logistica > precisión_RandomForest and precisión_logistica > precisión_Naive:
    mejorModelo = modelo_logistica
    mejorNombre = "Regresion Logistica"
    
print("Mejor modelo (modelo seleccionado):")
print(mejorNombre)
print("\n")
df["predicción_modelo"] = mejorModelo.predict(X)
#Creamos una nueva columna dentro del dataframe para almacenar los datos de predicción del modelo.
print("Ejemplo de canciones con prediccion:")
print(df[["track_name", "artists", "track_genre", "categoría", "predicción_modelo"]].head(20))

df_radio = df.sample(700)
#Aquí creamos un nuevo dataframe para la radio, a partir de 700 canciones aleatorias del dataframe original.
#Decidimos no fijar el random_state para que cada día que se genere la programación de 360 canciones, esta no sea idéntica y tenga variaciones.
programación = []
último_artista = ""
última_categoría = ""
canciones_usadas = []
for i, fila in df_radio.iterrows():
    nombre = fila["track_name"]
    artista = fila["artists"]
    duración = fila["duration_ms"]
    categoría = fila["predicción_modelo"]
    energía = fila["energy"]
    baile = fila["danceability"]
    popularidad = fila["popularity"]
    positividad = fila["valence"]
    if nombre in canciones_usadas:
        continue
    if artista == último_artista:
        continue
    if categoría == última_categoría:
        continue
    #Aquí decidimos imponer unas reglas sobre la programación para evitar que se repitan canciones, artistas y géneros_paraguas seguidos.
    if energía < 0.45 and baile < 0.55 and (0.3 < positividad < 0.6):
        horario = "Mañana"
    elif energía >= 0.45 and energía < 0.70 and positividad >= 0.6:
        horario = "Tarde"
    elif energía >= 0.70 or baile >= 0.70 and positividad <= 0.3:
        horario = "Noche"
    #Aquí finalmente imponemos las reglas de clasificación para el horario en el que se pueden reproducir las canciones de acuerdo a su energía y su bailabilidad.
    programación.append([horario,nombre,artista,categoría,popularidad,energía,baile,valencia])
    canciones_usadas.append(nombre)
    último_artista = artista
    última_categoría = categoría
    if len(programación) == 360:
        break
programacionDeRadio = pd.DataFrame(programación,columns=[ "horario","cancion","artista","duración","categoría","popularidad","energia","danceability","positividad"])
#Convertimos la lista de canciones seleccionadas para cada horario en un dataframe final y estructurado.
print("\nProgramación generada para la radio:")
print(programacionDeRadio)
print("\nCanciones para la mañana:")
print(programacionDeRadio[programacionDeRadio["horario"] == "Mañana"].head(10))
print("\nCanciones para la tarde:")
print(programacionDeRadio[programacionDeRadio["horario"] == "Tarde"].head(10))
print("\nCanciones para la noche:")
print(programacionDeRadio[programacionDeRadio["horario"] == "Noche"].head(10))
programacionDeRadio.to_csv("programacion_radio.csv", index=False)
#Aquí mostramos las primeras 10 canciones guardadas para cada horario del día, es decir, mañana, tarde y noche.
#También convertimos la programación en un archivo .csv para que el abuelo de Regina pueda leerlo en excel y utilizar para su estación.
