
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

path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
print(path)

df = pd.read_csv("/kaggle/input/-spotify-tracks-dataset/dataset.csv")
print(df.info())

df=df[df["popularity"]!=0]
print(df.head(10))
print(df.info())

#Aqui defininimos la función en la cual se agrupan los 114 generos del dataset en los 7 generos simplificados creados
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
      
df["Genero"] = df["track_genre"].apply(agrupar_porgenero)
print("\nCantidad de canciones por genero:")
print(df["Genero"].value_counts())

df = df[df["Genero"] != "otro"]
print(df.shape)
print("\nClasificacíon final de generos:")
print(df["Genero"].value_counts())

x=df[["popularity","duration_ms","danceability","energy","loudness","speechiness","acousticness","instrumentalness","liveness","valence","tempo"]]
y=df["Genero"]
x_entren, x_prueba, y_entren, y_prueba = train_test_split(x,y,test_size=0.25,random_state=42)
print("Datos de entrenamiento: ", X_train.shape)
print("Datos de prueba:", X_test.shape)

Naive_Bayes = GaussianNB()
Naive_Bayes.fit(x_entren, y_entren)
predicción_Naive = Naive_Bayes.predict(x_prueba)
precisión_Naive = accuracy_score(y_prueba, predicción_Naive)
print("Resultado de precisión de Naive-Bayes")
print(precisión_Naive)
print("\n")
print("Reporte de modelo Naive-Bayes:")
print(classification_report(y_test, predicción_nb))

modelo_lr = LogisticRegression(max_iter=1000)
modelo_lr.fit(X_train, y_train)
predicción_lr = modelo_lr.predict(X_test)
precisión_lr = accuracy_score(y_test, predicción_lr)
print("\nPrecisión Regresión Logística:")
print(precisión_lr)
print("\nReporte Regresión Logística:")
print(classification_report(y_test, predicción_lr))

modelo_rf = RandomForestClassifier(n_estimators=100,random_state=42,max_depth=20)
modelo_rf.fit(X_train, y_train)
predicción_rf = modelo_rf.predict(X_test)
precisión_rf = accuracy_score(y_test, predicción_rf)
print("\nPrecisión del Random Forest:")
print(precisión_rf)
print("\nReporte Random Forest:")
print(classification_report(y_test, predicción_rf))
print("\nComparación general:")
print("Naive Bayes:", precisión_nb)
print("Regresión Logística:", precisión_lr)
print("Random Forest:", precisión_rf)

mejor_modelo = modelo_rf
mejor_nombre = "Random Forest"
if precisión_nb > precisión_rf and precisión_nb > precisión_lr:
    mejor_modelo = modelo_nb
    mejor_nombre = "Naive Bayes"
if precisión_lr > precisión_rf and precisión_lr > precisión_nb:
    mejor_modelo = modelo_lr
    mejor_nombre = "Regresion Logistica"
print("\nMejor modelo:")
print(mejor_nombre)
df["predicción_modelo"] = mejor_modelo.predict(X)
print("\nEjemplo de canciones con prediccion:")
print(df[["track_name", "artists", "track_genre", "categoría", "predicción_modelo"]].head(20))

df_radio = df.sample(700, random_state=10)
programación = []
último_artista = ""
última_categoría = ""
canciones_usadas = []
for i, fila in df_radio.iterrows():
    nombre = fila["track_name"]
    artista = fila["artists"]
    categoría = fila["predicción_modelo"]
    energía = fila["energy"]
    baile = fila["danceability"]
    popularidad = fila["popularity"]
    valencia = fila["valence"]
    if nombre in canciones_usadas:
        continue
    if artista == último_artista:
        continue
    if categoría == última_categoría:
        continue
    if energía < 0.45 and baile < 0.55:
        horario = "Mañana"
    elif energía >= 0.45 and energía < 0.70:
        horario = "Tarde"
    elif energía >= 0.70 or baile >= 0.70:
        horario = "Noche"
    programación.append([horario,nombre,artista,categoría,popularidad,energía,baile,valencia])
    canciones_usadas.append(nombre)
    último_artista = artista
    última_categoría = categoría
    if len(programación) == 360:
        break
programacionDeRadio = pd.DataFrame(programación,columns=[ "horario","cancion","artista","categoría","popularidad","energia","danceability","valence"])
print("\nProgramacion generada para la radio:")
print(programacionDeRadio)
print("\nCanciones para la mañana:")
print(programacionDeRadio[programacionDeRadio["horario"] == "Mañana"])
print("\nCanciones para la tarde:")
print(programacionDeRadio[programacionDeRadio["horario"] == "Tarde"])
print("\nCanciones para la noche:")
print(programacionDeRadio[programacionDeRadio["horario"] == "Noche"])
print("\nCanciones para fin de semana:")
print(programacionDeRadio[programacionDeRadio["horario"] == "Fin de semana"])
programacionDeRadio.to_csv("programacion_radio.csv", index=False)
print("\nSe guardo el archivo programacion_radio.csv")
