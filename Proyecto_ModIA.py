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

def agrupar_genero(genero):
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
df["categoría"] = df["track_genre"].apply(agrupar_genero)
#Creamos una nueva columna llamada categoría, la cual contiene el género paraguas de cada canción.
print("\nCantidad por género:")
print(df["categoría"].value_counts())

df = df[df["categoría"] != "otro"]
#Eliminamos todos las filas que tuvieran como género paraguas otro, ya que es demasiada mezcla entre géneros no importantes para la estación de radio del abuelo.
#También consideramos importante eliminar estos datos porque pueden presentar demasiado ruido para nuestro modelo.
print("\nDataset después de quitar la clasificación de género otro:", df.shape)
print("\nGéneros finales:")
print(df["categoría"].value_counts())

X=df[["popularity","duration_ms","danceability","energy","loudness","speechiness",
      "acousticness","instrumentalness","liveness","valence","tempo"]]
y=df["categoría"]
#Aquí seleccionamos nuestras variables predictoras y nuesta variable objetivo. Seleccionamos 11 variables para predecir el género.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#Con más de 34000 datos consideramos que una separación de 80% - 20% de los datos para entrenamiento y prueba era suficiente para el modelo.
print("Datos de entrenamiento: ", X_train.shape)
print("Datos de prueba:", X_test.shape)

modelo_nb = GaussianNB()
modelo_nb.fit(X_train, y_train)
#Primero entrenamos un modelo basado en el teorema de Naive Bayes, el cual supone que todos las variables son independientes entre sí.
predicción_nb = modelo_nb.predict(X_test)
precisión_nb = accuracy_score(y_test, predicción_nb)
print("\nPrecisión Naive Bayes:", precisión_nb)
print("\nReporte Naive Bayes:")
print(classification_report(y_test, predicción_nb))

modelo_lr = LogisticRegression(max_iter=10000)
modelo_lr.fit(X_train, y_train)
#Aquí entrenamos un modelo lineal, el cual mide la probabilidad de pertenecer a cada género paraguas.
#Debido a la cantidad de datos decidimos utilizar max_iter=10000 para asegurar que haya convergencia.
predicción_lr = modelo_lr.predict(X_test)
precisión_lr = accuracy_score(y_test, predicción_lr)
print("\nPrecisión Regresión Logística:", precisión_lr)
print("\nReporte Regresión Logística:")
print(classification_report(y_test, predicción_lr))

modelo_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=1,
    max_depth=20
)
#Finalmente, entrenamos un modelo de random forest con 100 árboles de decisión, una profundidad máxima de cada árbol de 20 nodos,
#ya que es un valor intermedio y evita el overfitting. El random state simplemente lo fijamos a 1 para asegurar que los resultados no cambien.
modelo_rf.fit(X_train, y_train)
predicción_rf = modelo_rf.predict(X_test)
precisión_rf = accuracy_score(y_test, predicción_rf)
print("\nPrecisión del Random Forest:", precisión_rf)
print("\nReporte Random Forest:")
print(classification_report(y_test, predicción_rf))
print("\nComparación general:")
print("Naive Bayes:", precisión_nb)
print("Regresión Logística:", precisión_lr)
print("Random Forest:", precisión_rf)
#Aquí finalmente comparamos la precisión de los 3 modelos para elegir cuál es el mejor.

#Observando la precisión de los 3 modelos, concluimos que random forest es el mejor modelo a utilizar y es el que finalmente ocupamos para nuestra predicción.
df["predicción_modelo"] = modelo_rf.predict(X)
#Creamos una nueva columna dentro del dataframe para almacenar los datos de predicción del modelo.
print("\nEjemplo de dataframe con predicciones ya incluidas:", df.head(20))

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
    programación.append(
        [horario,nombre,artista,duración,categoría,popularidad,energía,baile,positividad]
    )
    canciones_usadas.append(nombre)
    último_artista = artista
    última_categoría = categoría
    if len(programación) == 360:
        break
programacionDeRadio = pd.DataFrame(
    programación,
    columns=[ "horario","canción","artista","duración","categoría","popularidad","energía","baile","positividad"]
)
#Convertimos la lista de canciones seleccionadas para cada horario en un dataframe final y estructurado.
print("\nProgramación generada para la radio:")
print("Canciones para la mañana:")
print(programacionDeRadio[programacionDeRadio["horario"] == "Mañana"].head(10))
print("\nCanciones para la tarde:")
print(programacionDeRadio[programacionDeRadio["horario"] == "Tarde"].head(10))
print("\nCanciones para la noche:")
print(programacionDeRadio[programacionDeRadio["horario"] == "Noche"].head(10))
programacionDeRadio.to_csv("programación_radio.csv", index=False)
print("\nSe guardó el archivo programación_radio.csv")
#Aquí mostramos las primeras 10 canciones guardadas para cada horario del día, es decir, mañana, tarde y noche.
#También convertimos la programación en un archivo .csv para que el abuelo de Regina pueda leerlo en excel y utilizar para su estación.
