# Nesterov's Accelerated Gradient for Adam-type Learning Algorithms

A technique that accelerates Adam-type optimizers, like Adam and AMSGrad, for training neural networks. We demonstrated this technique by incorporating it into Adam, AMSGrad and AdaMax. However, it also works on other algorithms, like NosAdam and Adabound, as long as they utilize first-order moment to accelerate convergence. 

This repository includes codes, based on tensorflow framework, for new optimizers and experiments. New optimizers can be used when 'MyOptimizers.py' file is imported,
```python3
from MyOptimizers import AdamPlus, AMSGradPlus, AdaMaxPlus
```
and to generate replicable results shown in the paper, 'Cifar10.py' and 'MNIST_autoencoder.py' have to be run in tensorflow environment.

```
python Cifar10.py
python MNIST_autoencoder.py
```
