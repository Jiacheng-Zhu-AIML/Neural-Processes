# Neural-Processes
This repository contains PyTorch implementations of the following Neural Process variants


*   Neural Processes (NPs)
*   Attentive Neural Processes (ANPs)
*   ANPRNN

The model architectures follow the ones proposed in the papers.

## Installation & Requirements
It is recommended to use Python3. Using virtual environment is recommended.

```
PyTorch == 1.4.0
numpy
...
```
To run the example notebooks, please install all requirements:
```
pip3 install -r requirements.txt
```

## Descriptions
*   The Neural Process models are under ```/neural_process_models``` folder
*   In ```/neural_process_models/modules``` there are all the modules for building NP networks, 
including linear MLP, attention network and so on.
*   The data generation functions are under ```\data```
*   Say something about the main scrips under root

## Results

#### 1d function regression
![10 context points](misc/images/anp_-3_3.gif?raw=true  "Title" )

#### Image condition reconstruction
![10 context points](misc/images/placeholder?raw=true  "Title")

## Exampels

Check 

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
## 

## License
MIT