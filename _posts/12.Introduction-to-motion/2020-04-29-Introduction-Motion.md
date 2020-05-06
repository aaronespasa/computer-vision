---
layout: post
title: "Teoría"
category: motion-introduction
author: aaron
---

{% assign imgUrl = "/assets/12.Introduction-to-motion/" | prepend: site.baseurl%}

# Introducción al movimiento

Localización: Buscar dónde se encuentra un objeto en movimiento con cierta precisión. Se utiliza por ejemplo en vehículos autónomos.

### Movimiento

Las mismas técnicas aplicadas a imágenes en las pasadas secciones puedes ser aplicadas a vídeos ya que estos son un conjunto de ellas (denominadas frames).

Pero lo que distingue principalmente a un vídeo de una imágen es la idea de movimiento.

Una forma de localizar objetos en el tiempo y detectar su movimiento es extrayendo ciertos features y observando cómo cambian de un frame a otro.

#### Optical Flow

Es utilizado en aplicaciones de seguimiento y análisis de movimiento.

Funciona asumiendo dos cosas sobre los frames de la imagen:

1. La intensidad de los píxeles se mantiene constante entre frames
2. Los píxeles cercanos tienen un movimiento similar.

El seguimiento de un punto proporciona información sobre la velocidad de movimiento y datos que pueden ser utilizados para predecir la futura localización del punto.

Por esto, podemos utilizar el optical flow para aplicaciones como:

- Reconocimiento de gestos con las manos.
- Monitorear el movimiento de un vehículo.
- Distinguir si una persona está caminando o corriendo.
- Predecir el movimiento de un vehículo para así evitar obstáculos.
- Seguimiento de ojos para VR.

<br/>

Si tenemos dos imágenes y queremos saber cómo se ha movido un determinado punto de una de ellas, lo que hacemos es calcular el vector de movimiento que describa la velocidad de este punto desde el primer frame hasta el siguiente.

<img src="{{ "motion-vector.png" | prepend: imgUrl }}" class="md_image"/>

<p style="text-align:center">Vector de movimiento - Imagen de Udacity</p>

Si nos fijamos en el punto del primer frame, para calcular un segundo punto lo único que debemos es trazar un vector hacia él. De esta forma, las coordenadas del segundo punto serán las del primero más el vector.

También podríamos hallar su magnitud con el teorema de pitágoras y el ángulo del triángulo que forman con trigonometría, utilizando la fórmula de la tangente.

##### Primera suposición: Constancia del brillo

Teniendo en cuenta que el brillo no va a presentear la ecuación, podemos formular la siguiente ecuación donde I es la intensidad lumínica y t el tiempo:


$$
I(x, y, t) = I(x + u, y + v, t + 1)
$$


<p style="text-align:center">Suposición de la constancia del brillo</p>



Esta función puede ser descompuesta en una expansión de las Series de Taylor:


$$
I(x, y, t) = \frac{\partial I}{\partial x}u + \frac{\partial I}{\partial y}v \frac{\partial I}{\partial t}1 + I(x, y, t)
$$


Vemos que podemos eliminar I(x, y, t) y entonces podemos relacion las cantidades de los vectores de movimieto u y v con el tiempo:


$$
\frac{\partial I}{\partial x}u + \frac{\partial I}{\partial y}v = -\frac{\partial I}{\partial t}
$$


Y estos son los fundamentos de cómo optical flow estima los vectores de movimiento para un set de feature points en un vídeo.

##### Segunda suposición: Similitud de movimiento en píxeles cercanos

Suponemos que píxeles cercanos tendrán un movimiento parecido.

Matemáticamente esto significa que los píxeles en una determinada área tienen motion vectors similares.

Por ejemplo, pensando en una persona, si tú estás monitoreando un conjunto de puntos de su cara, todos esos puntos deberían moverse a la misma velocidad. Tu nariz no puede moverse hacia el lado opuesto que tu barbilla.

<br/>

Un caso en el que se utilice el optical flow para hacer seguimiento de los objetos que nos rodean en un vídeo es el [NVIDIA Redtail drone](https://blogs.nvidia.com/blog/2017/06/09/drone-navigates-without-gps/).