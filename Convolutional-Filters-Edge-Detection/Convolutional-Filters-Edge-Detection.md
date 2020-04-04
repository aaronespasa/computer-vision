# Filtros convolucionales y detección de bordes

### Intensidad de la escala de grises y frecuencia

Además de aprovechar la información que nos proporcionan los colores como vimos en la primera lección ("Image-Representation-Classification"), también podemos utilizar los patrones de la **intensidad de la escala de grises** para representar y clasificar una imagen.

La intensidad es una medida, al igual que la luminosidad, de lo claro y lo oscuro. Así, podemos detectar otras áreas u objetos de interés. Por ejemplo, podemos identificar los bordes de un objeto observando los grandes cambios producidos en la intensidad.

En el ejemplo presentado a continuación podemos ver ese salto de la intensidad respecto de las personas y el fondo.

![](/home/aaronespasa/Documents/computer-vision/Convolutional-Filters-Edge-Detection/Images/grayscale-example.png)

Esta **tasa de cambio** de la intensidad en el campo del computer vision se le denomina como **frecuencia**. De esta forma, los componentes con una alta frecuencia también corresponden a los bordes de un objeto en las imágenes, lo que nos ayuda a clasificar a estos objetos.

Como podemos utilizar la **Transformada de Fourier** ([Introducción visual a la Transformada de Fourier](https://www.youtube.com/watch?v=spUNpyF58BY)) para extraer las la frecuencias del sonido, también la podemos utilizar como una herramienta de procesamiento de imágenes con el fin de descomponerla en sus componentes de frecuencia como hemos visto anteriormente. Para hacer esto, la Transformada de Fourier (a partir de ahora, FT por sus siglas en inglés) trata los patrones de intensidad en una imagen como ondas senoidales con una determinada frecuencia ([Artículo sobre las transformadas de fourier en imágenes](https://plus.maths.org/content/fourier-transforms-images)]).

En el sonido varía la intensidad sonora con respecto al tiempo. Haciendo una comparación con el sonido, en una imagen el tiempo sería la localización del píxel y la intensidad sería en este caso la de la escala de grises yendo del 0 al 255.

![](/home/aaronespasa/Documents/computer-vision/Convolutional-Filters-Edge-Detection/Images/image-ft.jpg)

Si quieres ver cómo aplicarlas con OpenCV échale un ojo al archivo "fourier-transforms.ipynb".

## Filtros High-pass

En una imagen, los filtros son utilizados para:

1. Filtrar la información que no queremos.
2. Aumentar características de interés (como los límites de un objeto).

Los **filtros high-pass** nos permiten agudizar una imagen para así poder mejorar las partes con altas frecuencias de la imagen. Para entender cómo funcionan estos filtros, necesitamos utilizar los **convolutional kernels** (un kernel es simplemente una matriz de números que llamamos ponderaciones que modifica una imagen). Con ellos podemos realizar un filtro para detectar bordes si conseguimos que la suma de todas las filas sea 0. Debe dar 0 porque este filtro está computando la diferencia entre los píxeles cercanos. Si la suma no fuese 0, nos daría imágenes con más luminosidad (<0) o más oscuras (>0).

Para realizar un filtro que nos permita detectar bordes, una imagen de entrada (F(x,y)) es "convolucionada" con el kernel. (Esa **convolución** consiste en tomar «**grupos de píxeles cercanos»** de la imagen de entrada e ir realizando el producto escalar en el kernel.). Para verlo visualmente: [CNN Visualization](https://www.youtube.com/watch?time_continue=19&v=f0t-OCG79-U&feature=emb_title)

![](/home/aaronespasa/Documents/computer-vision/Convolutional-Filters-Edge-Detection/Images/convolutional-kernels.png)

​																Imagen de Udacity



Para ver cómo realizar un filtro de detección de bordes con OpenCV recomiendo visitar el archivo "edge-detection-filter.ipynb".

### Gradientes

Los gradientes son una medida del cambio de intensidad en una imagen y, generalmente, marcan los límites de un objeto. Si pensamos en una imagen como una función (al igual que antes al realizar la convolución), podemos pensar en su gradiente como la derivada de esta función (F'(x,y)), la cual indica la medida del cambio de intensidad.

### Sobel filters

Al aplicar un filtro sobel a una imagen conseguimos coger una aproximación de la derivada de la imagen en la dirección x o y. Este tipo de filtro también detecta la magnitud (que tan intenso es el borde), para conseguir esto simplemente tenemos que calcular la raíz cuadrada del sobel (en la dirección que deseemos) al cuadrado.

Calculando la dirección del gradiente de la imagen en las direcciones x e y por separados podemos determinar la dirección del gradiente. La fórmula de este es la inversa de la tangente de la división de sobel e la dirección y entre el sobel en la dirección x.

## Filtros low-pass

Uno de los problemas que podemos ver en el archivo "edge-detection-filter.ipynb" es que al aplicar el filtro high-pass obtenemos una imagen con un poco de ruido. Para solucionar el problema, los filtros low-pass nos ayudan a desenfocar y suavizar la imagen, además de bloquear partes con una alta frecuencia de la imagen.

Para reducir este ruido podemos coger una media de los píxeles que rodean a otro para que así no se produzcan grandes saltos en la intensidad, especialmente en las áreas pequeñas. A esto se le conoce como **filtros low-pass**.

La diferencia con un filtro high-pass es que el low-pass suele tomar un promedio de las ponderaciones (ej. una matriz 3x3 de todo 1s) en vez de una diferencia como hacíamos con el filtro high-pass.

En el caso de una matriz 3x3 de 1s, la suma de todos sus elementos sería 9, por lo que tenemos que normalizarla dividiéndola por la cantidad de elementos que hay, dando un resultado así de 1.

### Desenfoque gaussiano (El filtro low-pass más utilizado en computer vision)

Es utilizado para preservar mejor los bordes de la imagen. Consiste principalmente en unas ponderaciones equilibradas que dan el mayor peso al píxel central.

Ej.:

1/16 * np.array([[1, 2, 1],

​				 			[2, 4, 2],

​				 			[1, 2, 1]])

Para más información ver el archivo "gaussian-blur.ipynb"



Tanto los filtros high-pass como low-pass son lo que más tarde definirá el comportamiento de las redes neuronales convolucionales.



## Canny Edge Detector

Sin embargo, aún no hemos terminado de determinar el cambio del nivel de intensidad que constituye un borde y cómo podemos representar representar bordes más gruesos o más finos.

Uno de los detectores de bordes más famosos para solucionar este problema es el Detector de Bordes Canny. Este es muy usado porque lleva a cabo una serie de pasos que produce con frecuencia  bastante precisión al detectar bordes. Esto lo realiza mediante los siguientes pasos:

1. Filtra el ruido utilizando el desenfoque gaussiano.

2. Busca la intensidad y la dirección de los bordes usando filtros Sobel.

3. Utilizando el output de los filtros Sobel, aplica una non-maximum suppression (NMS) para aislar los bordes más fuertes y hacerlos más finos hasta llegar a crear líneas con una anchura de 1px.

4. Utiliza un proceso llamado "hysteresis" para seleccionar los mejores bordes. "Hysteresis" consiste en realizar un doble proceso de umbral. Visto de una manera más visual:

   

   ![](/home/aaronespasa/Documents/computer-vision/Convolutional-Filters-Edge-Detection/Images/canny-edge-detection.png)
   
   ​																			Imagen de Udacity
   
   

Para más información, ver el archivo "canny-edge-detection.ipynb"

## Hough Transform

Trasforma los datos de la imagen de un sistema de coordenadas x,y en un espacio de Hough.

Una línea en el espacio imagen (sist. de coord. x,y) se representaría en el espacio de Hough con un punto.

![](/home/aaronespasa/Documents/computer-vision/Convolutional-Filters-Edge-Detection/Images/hough-space.png)

​																Imagen de Udacity



Para más información, visitar el archivo "hough-detections.ipynb"

## Haar Cascade Classifier ([Paper 2001](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf))

Puede ser interpretado como un acercamiento al machine learning en el que una función cascada es entrenada para resolver un problema de clasificación binaria. Después de que el clasificador vea que una imagen es una cara o no, este extrae las características de la imagen.

![](/home/aaronespasa/Documents/computer-vision/Image-Representation-Classification/Images/haar-cascade-classifier.gif)

​																	Imagen de Udacity

Funciona de la siguiente manera:

1. De una imagen detecta los haar features. Y los haar features detectan patrones como bordes, líneas y patrones rectangulares. Estos son muy importantes ya que diferentes secciones en las que se alterna la luminosidad y la oscuridad son las que definen una gran cantidad de features de nuestra cara.

2. El siguiente paso es una serie de clasificadores en cascada en la que en cada paso va aplicando un Haar feature detector:  De esta forma va descartando secciones de la imagen (De una foto de una cara como la de arriba estos serían el fondo y los hombros). Tras cada ronda, va realizando la clasificación y si no recibe suficiente información como para obtener una respuesta de la detección de features, entonces vuelve a aplicar el Haar feature detector. De ahí el nombre de un clasficador en cascada.

   De esta manera, el Haar cascade únicamente se centra en procesar y clasificar las áreas de la imagen que han sido clasificadas ya como parte de la cara, haciendo a este algoritmo muy veloz. 

Para más información, ver el archivo "haar-cascade-face-detection.ipynb".



Artículos y papers interesantes (Relacionados con la justicia, eliminar el bias, ...):

[Fairness in Machine Learning with PyTorch](https://godatadriven.com/blog/fairness-in-machine-learning-with-pytorch/)

[Delayed Impact of Fair Machine Learning](https://bair.berkeley.edu/blog/2018/05/17/delayed-impact/)

[Can We Keep Our Biases from Creeping into AI?](https://hbr.org/2018/02/can-we-keep-our-biases-from-creeping-into-ai?utm_campaign=hbr&utm_source=twitter&utm_medium=social)

[TED Talk: How I'm fighting bias in algorithms](https://www.ted.com/talks/joy_buolamwini_how_i_m_fighting_bias_in_algorithms)

[Gender Shades:  Intersectional Accuracy Disparities inCommercial Gender Classification](https://video.udacity-data.com/topher/2018/June/5b2c01ba_gender-shades-paper/gender-shades-paper.pdf)



