---
layout: post
title: "Teoría"
category: simultaneous-localization-mapping
author: aaron
---

{% assign imgUrl = "/assets/18.Simultaneous-Localization-Mapping/" | prepend: site.baseurl%}

<img src="{{ "cnn-rnn-representation.png" | prepend: imgUrl }}" class="md_image"/>

<p style="text-align:center">Representación de la RNN y CNN - Imagen de Udacity</p>

# SLAM: Simultaneous Localization And Mapping

La clave de por qué creamos mapas en los que el robot pueda moverse es porque este pierde el seguimiento de donde está debido a la incertidumbre del movimiento. Esto lo resolvimos en secciones anteriores dotándole a nuestro robot de un mapa, pero ¿y si no tenemos un mapa para proporcionarle al robot?

Aquí entra juego SLAM que nos permite recolectar información de los sensores y el movimiento del robot con el paso del tiempo y usar esa información para reconstruir un mapa del mundo.

Gráfico SLAM:

