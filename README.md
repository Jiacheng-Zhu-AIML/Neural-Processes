# Neural-Processes
This repository contains PyTorch implementations of the following Neural Process variants

*   Recurrent Attentive Neural Process (ANPRNN)
*   Neural Processes (NPs)
*   Attentive Neural Processes (ANPs)


The model architectures follow the ones proposed in the papers.
*   Marta Garnelo, et al. ["Neural Process"](https://arxiv.org/pdf/1807.01622.pdf) ICML 2018 Workshop.
*   Hyunjik Kim, et al. ["Attentive Neural Processes"](https://openreview.net/pdf?id=SkE6PjC9KX) ICLR 2019. 

This is the author implementation of NeurIPS 2019 Workshop paper
*   Shenghao Qin, Jiacheng Zhu, et al. ["Recurrent Attentive Neural Process for Sequential Data"](https://arxiv.org/abs/1910.09323)


## Installation & Requirements
It is recommended to use Python3. Using virtual environment is recommended.

```
PyTorch 
Numpy
Matplotlib
```

## Descriptions
*   The Neural Process models are under ```/neural_process_models``` folder
*   In ```/neural_process_models/modules``` there are all the modules for building NP networks, 
including linear MLP, attention network and so on.
*   The data generation functions are under ```\misc```

## Results

#### 1d function regression
![10 context points](misc/images/anp_-3_3.gif?raw=true  "Title" )



## Usage
Simple example of training an Attentive Neural Process of 1d function regression
```
python3 main_ANP_1d_regression.py
```

For digital number inpainting trained on MNIST data
```
python3 main_ANP_mnist.py
```
See Neural Process models in ```/neural_process_models``` for detailed examples of construction
 NP models for specific tasks. 


#### Acknowledgements
For any question, please refer to
*   Eric Ma (yingchem@umich.edu)
*   Jiacheng Zhu (jzhu4@andrew.cmu.edu)


## License
MIT