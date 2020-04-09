---
layout: post
title: "Teoría"
category: cnn-layers-feature-visualization
author: aaron
---
{% assign imgUrl = "/assets/5.CNN-Layers-Feature-Visualization/" | prepend: site.baseurl%}

# CNN Layers and Feature Visualization



## Convolutional Neural Networks (CNN)

Este tipo de red neuronal se encarga de encontrar y representar patrones en un espacio en 3D.

Una peculiaridad de esta red es que, en vez de enfocarse en los valores de los píxeles individualmente, se enfoca en un grupo de píxeles de un área de una imagen y aprende en ella los patrones espaciales.

{{ 'CNN-Layers.png' | asset_url | img_tag }}

<img src="{{ "CNN-Layers.png" | prepend: imgUrl }}" class="md_image"/>

<p style="text-align:center">Imagen de @cezannec (GitHub)</p>

