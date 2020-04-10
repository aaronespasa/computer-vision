---
layout: post
title: "Teoría"
category: cnn-layers-feature-visualization
author: aaron
---
{% assign imgUrl = "/assets/5.CNN-Layers-Feature-Visualization/" | prepend: site.baseurl%}

# CNN Layers and Feature Visualization



## Convolutional Neural Networks (CNN)

Este tipo de red neuronal se encarga de encontrar y representar patrones en un espacio en 3D y es la más potente a la hora de realizar tareas de procesamiento de imágenes.

Una peculiaridad de esta red es que, en vez de enfocarse en los valores de los píxeles individualmente, se enfoca en un grupo de píxeles de un área de la imagen gracias a las convoluciones (<a href="{{ site.baseurl }}/convolutional-filters-edge-detection/Convolutional-Filters-Edge-Detection">Sección 2</a>) que realiza y aprende de ella sus patrones característicos.

A medida que entrenamos la red, esta va actualizando las ponderaciones que definen cómo se filtra la imagen utilizando el backpropagation

<img src="{{ "CNN-Layers.png" | prepend: imgUrl }}" class="md_image"/>

<p style="text-align:center">Imagen de @cezannec (GitHub)</p>

Cada capa es representada como un paralelepípedo de distinto color.

###### 1ª Capa: **Convolutional layer**:

- Toma una imagen como input.
- Es una capa ya que está formada por una serie de convoluciones, en las cuales extrae una determinada característica. Por ejemplo, un filtro high-pass (<a href="{{ site.baseurl }}/convolutional-filters-edge-detection/Convolutional-Filters-Edge-Detection">Sección 2</a>) de una convolución se encargaría de detectar los bordes del objeto.
- El output de una convolutional layer es un set de feature maps (o activation maps) que son las versiones filtradas de la imagen original.

Tras la convolutional layer se ejecuta la **función de activación ReLu** que hace un input "x" 0 cuando este es negativo o igual a 0 (x <= 0) y 1 cuando es mayor a 0 (x > 0). El trabajo de las funciones de activación es hacer más eficiente el backpropagation y que así sea más efectivo entrenar la red neuronal.

<img src="{{ "Convolutional-Layers.png" | prepend: imgUrl }}" class="md_image"/>

<p style="text-align:center">4 filtros o convoluciones y su respectivo feature map - Imagen de Udacity</p>

Lo más interesante de esto es que podemos volver a aplicar una convolutional layer sobre los feature maps que ya habíamos obtenido, buscando así patrones sobre los patrones encontrados de la capa anterior.

<br/>

###### 2ª Capa: **Pooling layer**:

Encargada de reducir el tamaño de la matriz de píxeles de la imagen para así acelerar el proceso.

<img src="{{ "maxpooling-layer.png" | prepend: imgUrl }}" class="md_image"/>

<p style="text-align:center">Maxpooling layer - Imagen de machinecurve.com</p>

<br/>

###### Última capa: **Fully-connected layer**:

Se encarga se pasar la matriz o tensor a un vector de tres dimensiones.

<br/>

### Arquitectura de una CNN en Pytorch:

```python
import torch.nn as nn
import torch.nn.fuctional as F
```

```python
class Net(nn.Module):
    
    def __init__(self, n_classes):
        super(Net, self).__init__()
        
        # 1 canal de la imagen de input (en escala de grises)
        # 32 feature maps como output
        # 5x5 convolutional kernel
        self.conv1 == nn.Conv2d(1, 32, 5)
        
        # Maxpooling layer
        # Kernel de tamaño 2 y paso de 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully-connected layer
        # 32*4 es el tamaño del input después de la pooling layer
        self.fcl = nn.Linear(32*4, n_classes)
        
    # Definir la feedforward propagation
    def forward(self, x):
        # Conv/ReLu + pool layers
        x = self.pool(F.relu(self.convl(x)))
        
        # Preparar para convertir los feature maps en feature vectors
        x = x.view(x.size(0), -1)
        
        # Linear layer
        x = F.relu(self.fcl(x))
        
        return x
```

```python
# Instanciar e imprimir la red
n_classes = 20 # Número de clases
net = Net(n_classes)
print(net)
```

























