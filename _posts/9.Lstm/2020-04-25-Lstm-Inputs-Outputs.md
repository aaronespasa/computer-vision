---
layout: post
title: Inputs y Outputs de una LSTM en código
category: lstm
author: aaron
---
{% assign imgUrl = "/assets/9.Lstm/Lstm-Inputs-Outputs_files/" | prepend: site.baseurl%}
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

%matplotlib inline

torch.manual_seed(2)
```

### Definiendo una LSTM simple


```python
input_dim = 4
hidden_dim = 3
# nn.LSTM(input_size, hidden_size, num_layers)
# :param input_size:  Number of inputs
# :param hidden_size: Size of the hidden state,
#                     a function which contains the weights and represent the long and short term memory component.
#                     This will be the number of outputs that each LSTM cell produces at each time step.
# :param num_layers:  Number of hidden LSTM layers. Default value: 1.
lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)

# Make 5 input sentences of 4 random values each
inputs_list = [torch.randn(1, input_dim) for _ in range(5)]
print('inputs: \n', inputs_list, end="\n\n")

# initialize the hidden state
# (1 layer, 1 batch_size, 3 outputs)
# first tensor is the hidden state, h0
# second tensor initializes the cell memory, c0
h0 = torch.randn(1, 1, hidden_dim)
c0 = torch.randn(1, 1, hidden_dim)

# step through the sequence one element at a time.
for i in inputs_list:
    # after each step, hidden contains the hidden state
    out, hidden = lstm(i.view(1, 1, -1), (h0, c0))
    print('out: \n', out)
    print('hidden: \n', hidden, end="\n\n")

```

    inputs: 
     [tensor([[-0.7739,  0.0496, -0.6174,  2.2406]]), tensor([[ 1.3194, -1.3208, -0.2356, -0.3233]]), tensor([[0.0477, 1.0690, 1.0751, 0.3943]]), tensor([[ 1.1772, -0.7236, -0.2669,  0.2690]]), tensor([[-0.5728,  2.2614, -0.5147,  0.5446]])]
    
    out: 
     tensor([[[-0.4678,  0.0041, -0.2471]]], grad_fn=<StackBackward>)
    hidden: 
     (tensor([[[-0.4678,  0.0041, -0.2471]]], grad_fn=<StackBackward>), tensor([[[-0.6262,  0.0332, -0.3358]]], grad_fn=<StackBackward>))
    
    out: 
     tensor([[[-0.3292, -0.2225,  0.4648]]], grad_fn=<StackBackward>)
    hidden: 
     (tensor([[[-0.3292, -0.2225,  0.4648]]], grad_fn=<StackBackward>), tensor([[[-0.4465, -0.5758,  0.6697]]], grad_fn=<StackBackward>))
    
    out: 
     tensor([[[-0.0103, -0.0617,  0.4570]]], grad_fn=<StackBackward>)
    hidden: 
     (tensor([[[-0.0103, -0.0617,  0.4570]]], grad_fn=<StackBackward>), tensor([[[-0.0241, -0.4090,  0.6000]]], grad_fn=<StackBackward>))
    
    out: 
     tensor([[[-0.3421, -0.1574,  0.4006]]], grad_fn=<StackBackward>)
    hidden: 
     (tensor([[[-0.3421, -0.1574,  0.4006]]], grad_fn=<StackBackward>), tensor([[[-0.4899, -0.5758,  0.5336]]], grad_fn=<StackBackward>))
    
    out: 
     tensor([[[ 0.0059, -0.0539,  0.0390]]], grad_fn=<StackBackward>)
    hidden: 
     (tensor([[[ 0.0059, -0.0539,  0.0390]]], grad_fn=<StackBackward>), tensor([[[ 0.0109, -0.5090,  0.0439]]], grad_fn=<StackBackward>))



### Procesar todos los inputs a la misma vez
Procedimiento:
1. Concatenar todas nuestras secuencias de inputs e un único gran tensor, con un batch_size definido
2. Definir el tamaño de nuestro hidden state
3. Obtener los outputs y los hidden state más recientes


```python
# turn inputs into a tensor with 5 rows of data
# add the extra 2nd dimension (1) for batch_size
inputs = torch.cat(inputs_list).view(len(inputs_list), 1, -1)

# print out our inputs and their shape
# you should see (number of sequences, batch size, input_dim)
print('inputs size: \n', inputs.size(), end="\n\n")
print('inputs: \n', inputs, end="\n\n")

# initialize the hidden state
h0 = torch.randn(1, 1, hidden_dim)
c0 = torch.randn(1, 1, hidden_dim)

# get the outputs and hidden state
out, hidden = lstm(inputs, (h0, c0))

print('out: \n', out, end="\n\n")
print('hidden: \n', hidden)
```

    inputs size: 
     torch.Size([5, 1, 4])
    
    inputs: 
     tensor([[[-0.7739,  0.0496, -0.6174,  2.2406]],
    
            [[ 1.3194, -1.3208, -0.2356, -0.3233]],
    
            [[ 0.0477,  1.0690,  1.0751,  0.3943]],
    
            [[ 1.1772, -0.7236, -0.2669,  0.2690]],
    
            [[-0.5728,  2.2614, -0.5147,  0.5446]]])
    
    out: 
     tensor([[[-0.3707,  0.0372, -0.0486]],
    
            [[-0.2980, -0.1198,  0.2102]],
    
            [[-0.0199, -0.0067,  0.4572]],
    
            [[-0.0158, -0.0862,  0.4438]],
    
            [[ 0.0961, -0.0342,  0.2408]]], grad_fn=<StackBackward>)
    
    hidden: 
     (tensor([[[ 0.0961, -0.0342,  0.2408]]], grad_fn=<StackBackward>), tensor([[[ 0.4994, -0.2241,  0.3038]]], grad_fn=<StackBackward>))



```python

```
