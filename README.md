# Sistema inteligente para clasificación y programación musical en una estación de radio

## Descripción del proyecto

Este proyecto tiene como finalidad aplicar un modelo de Machine Learning a un problema real dentro de una estación de radio. La idea surge a partir de la necesidad de organizar mejor la música disponible, ya que cuando se tienen muchas canciones puede ser complicado mantener una programación variada, separar horarios por estilo musical y evitar que ciertos artistas o canciones se repitan demasiado.

Para esto, se utilizó un dataset de canciones con información musical como popularidad, duración, energía, bailabilidad, tempo, valence y género. A partir de estos datos, se entrenaron distintos modelos de clasificación para agrupar canciones en categorías musicales generales. Finalmente, el modelo seleccionado se utilizó para generar una programación de radio con 360 canciones.

## Integrantes

- Regina Franco Gutierrez - A01352605
- Ezra Daniel Ruiz Arredondo - A01661151
- Luis López Horcasitas - A01713551

## Objetivo

Desarrollar un sistema básico de clasificación musical que permita organizar canciones por categorías generales y apoyar la creación de una programación de radio más ordenada, variada y menos repetitiva.

## Dataset utilizado

El archivo utilizado fue:

```text
Spotify.csv
```

El dataset contiene canciones con distintas características musicales. Para este proyecto se usaron principalmente las siguientes columnas:

```text
track_name
artists
popularity
duration_ms
danceability
energy
loudness
speechiness
acousticness
instrumentalness
liveness
valence
tempo
track_genre
```

Estas variables permiten representar cada canción de forma numérica, lo cual facilita que los modelos puedan aprender patrones y clasificar canciones según sus características.

## Limpieza de datos

Antes de entrenar los modelos, se realizó una limpieza básica del dataset. Primero se eliminaron datos duplicados y filas con valores vacíos. Después, se seleccionaron únicamente las columnas necesarias para el modelo.

Asimismo, se omitieron las canciones con popularidad igual a cero, ya que para una programación de radio estas canciones pueden no aportar tanta relevancia. También se eliminaron los géneros que quedaron dentro de la categoría `otro`, porque agrupaban estilos muy distintos entre sí.

## Agrupación de géneros

Como el dataset original contiene muchos géneros musicales, se agruparon en categorías más generales. Esto permitió que el problema fuera más claro y fácil de trabajar.

Las categorías utilizadas fueron:

```text
pop_comercial
rock
latino
electronica
tranquila
urbana
```

Por ejemplo, géneros como `pop`, `dance` y `disco` se agruparon como `pop_comercial`. De igual manera, géneros como `rock`, `alt-rock`, `punk` y `metal` se agruparon dentro de `rock`.

Esta agrupación fue importante porque ayudó a reducir la cantidad de clases y permitió que el modelo trabajara con categorías más representativas para una estación de radio.

## Modelos utilizados

Se entrenaron y compararon tres modelos de clasificación:

```text
Naive Bayes
Regresión Logística
Random Forest
```

La comparación se realizó para observar cuál modelo tenía mejor desempeño al momento de clasificar canciones. Si bien los tres modelos permiten resolver problemas de clasificación, Random Forest suele ser útil cuando se trabaja con varias características numéricas, como ocurre en este dataset.

## Variables del modelo

Las variables de entrada fueron:

```text
popularity
duration_ms
danceability
energy
loudness
speechiness
acousticness
instrumentalness
liveness
valence
tempo
```

La variable que el modelo debía predecir fue:

```text
categoria
```

Es decir, el modelo aprende a clasificar una canción dentro de una categoría musical general usando sus características numéricas.

## Entrenamiento y prueba

Los datos se dividieron en entrenamiento y prueba usando:

```python
test_size=0.25
random_state=42
```

Esto significa que el 75% de los datos se usó para entrenar el modelo y el 25% restante para probarlo. El valor `random_state=42` se utilizó para que la separación de datos fuera igual cada vez que se ejecuta el código.

## Evaluación

Para evaluar los modelos se utilizaron tres elementos principales:

```text
Accuracy
Classification report
Matriz de confusión
```

El accuracy permite conocer el porcentaje general de aciertos. El classification report muestra métricas más detalladas como precision, recall y f1-score. Por otro lado, la matriz de confusión ayuda a observar en qué categorías el modelo se equivoca con mayor frecuencia.

## Generación de programación de radio

Después de elegir el mejor modelo, se generó una programación musical con 500 canciones. Para esto, el código toma una muestra grande del dataset y aplica reglas simples para evitar una programación repetitiva.

Las reglas utilizadas fueron:

```text
No usar canciones con popularidad cero.
No repetir una canción ya utilizada.
No repetir el mismo artista de forma consecutiva.
No repetir la misma categoría musical de forma consecutiva.
Asignar un horario según energía y bailabilidad.
```

Los horarios generados fueron:

```text
Mañana
Tarde
Noche
Fin de semana
```

La asignación de horarios se hizo de forma sencilla. Las canciones con menor energía y menor bailabilidad se asignaron a la mañana. Las canciones con energía media se asignaron a la tarde. Las canciones con mayor energía o mayor bailabilidad se asignaron a la noche. Finalmente, las canciones que no entraban claramente en esos casos se asignaron a fin de semana.

## Archivo generado

Al ejecutar el código, se genera el archivo:

```text
programacion_radio.csv
```

Este archivo contiene la programación final con las siguientes columnas:

```text
horario
cancion
artista
categoria
popularidad
energia
danceability
valence
```

## Estructura del proyecto

La estructura del repositorio puede organizarse de la siguiente forma:

```text
proyecto-radio-ml/
│
├── dataset.csv
├── main.py
├── programacion_radio.csv
├── requirements.txt
├── README.md
└── DU-IAG.md
```

El archivo `programacion_radio.csv` se genera automáticamente después de ejecutar el programa.

## Dependencias

Para ejecutar el proyecto se necesitan las siguientes librerías:

```text
pandas
numpy
scikit-learn
```

El archivo `requirements.txt` debe contener:

```text
pandas
numpy
scikit-learn
```

## Instrucciones de ejecución

Primero, se deben instalar las dependencias:

```bash
pip install -r requirements.txt
```

Después, se ejecuta el archivo principal:

```bash
python main.py
```

Al finalizar, el programa mostrará en consola el tamaño del dataset, la cantidad de canciones por categoría, los resultados de cada modelo, el mejor modelo seleccionado y la programación generada.

## Resultados esperados

El resultado esperado es obtener un modelo capaz de clasificar canciones en categorías musicales generales y generar una programación de radio de 500 canciones. Esta programación busca mantener variedad, evitar repeticiones directas y ordenar las canciones según características como energía y bailabilidad.

## Aporte del proyecto

El aporte principal del proyecto no está solamente en entrenar un modelo de clasificación, sino en conectar ese modelo con una necesidad práctica. En este caso, el sistema ayuda a organizar canciones para una estación de radio, tomando en cuenta criterios musicales y reglas básicas de programación.

De esta forma, el proyecto muestra cómo Machine Learning puede funcionar como una herramienta de apoyo para tomar decisiones más ordenadas dentro de un contexto real.

## Limitaciones

El sistema tiene algunas limitaciones. La agrupación de géneros se realizó de forma manual, por lo que podría mejorarse con más análisis musical. Asimismo, la programación generada no considera todavía datos reales de audiencia, horarios específicos de transmisión o preferencias de los locutores.

No obstante, el sistema cumple con el objetivo principal: clasificar canciones y generar una programación básica, funcional y menos repetitiva.

## Posibles mejoras

Algunas mejoras futuras serían:

```text
Usar letras o títulos de canciones con procesamiento de texto.
Aplicar validación cruzada.
Optimizar hiperparámetros.
Agregar más reglas de programación.
Crear una interfaz para cargar canciones nuevas.
Filtrar canciones por idioma, artista o popularidad mínima.
Generar gráficas para explicar los resultados.
```

## Conclusión

Este proyecto muestra una aplicación sencilla pero funcional de Machine Learning en un contexto real. A través del análisis de características musicales, el sistema clasifica canciones y genera una programación de radio más ordenada. Si bien todavía puede mejorarse, permite observar el proceso completo de un proyecto de aprendizaje automático: limpieza de datos, entrenamiento, evaluación, selección del modelo y aplicación práctica.
