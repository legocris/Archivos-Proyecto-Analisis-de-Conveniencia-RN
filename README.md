## English
# NN-Comparitive-Analysis-Project-Files
Files used to make the Capstone Project on the Comparative Analysis of Shallow vs Deep Neural Networks using the Perceptron for Classification Problems. 

The project was submitted as the **Mathematical Modeling Capstone Project for the Bachelor's Degree in Mathematics. CUCEI. UDG**.

In the thesis, the performance of various neural network architectures is tested on different classification problems. The architectures vary but maintain a constant number of neurons; this serves as a "fairness" parameter in order to make statistical comparisons of the results of each architecture on the same classification problem.

The project can be found in [ProyectoIntegrador[en].pdf](/ProyectoIntegrador[en].pdf).

# Files
<img width="321" height="527" alt="imagen" src="https://github.com/user-attachments/assets/0c1bbd42-4731-4895-89ff-40c3709bc8f3" />

Each folder includes the Preprocessed data for a classification problem. The data located in the `/Bin` sub-folder is fed into the `Red.exe` program, where it is processed by neural networks with varying numbers of layers and neurons using the Backpropagation algorithm to solve each classification problem. The training results for each neural network are exported to the `Reg` folder, to finally be processed by a Python script, `analisis_de_registros.py`, which converts them into readable `.xlss` files and `.png` graphs.

To obtain the datasets and preprocess them for subsequent training in CUDA, the script "generate_cuda_datasets.py" was used.
The main file that trains the neural networks is "Red.cu", it implements the Backpropagation algorithm using CUDA BLAS API.

<img width="1600" height="700" alt="imagen" src="https://github.com/user-attachments/assets/7efcf2aa-c0bc-4c47-bccc-3c32ed82c44e" />

Example of the comparison for the solution to the Spiral problem by 9-neuron neural networks, but with different architectures: one with 1 hidden layer of 9 neurons and another with 3 hidden layers, each with 3 neurons. The results are favorable for the Narrow Network, as it was able to solve the classification problem in every single attempt, whereas the deep network was only capable of doing so in one of them.

---

## Español
# Archivos-Proyecto-Analisis-de-Conveniencia-RN
Archivos usados para hacer el Proyecto Integrador de Análisis de Convergencia de Redes Neuronales Angostas contra Profundas usando el perceptrón para problemas de clasificación. 

El proyecto fue entregado como **Proyecto Integrador de Modelación Matemática en la Licenciatura en Matemáticas. CUCEI. UDG**.

En la tesis se prueba el desempeño de diversas arquitecturas de redes neuronales sobre diversos problemas de clasificación. Las arquitecturas son variadas pero mantienen una constante en número de neuronas, esto como parámetro de "justicia" y hacer comparaciones estadísticas de los resultados de cada arquitectura sobre el mismo problema de clasificación.

El proyecto está en [ProyectoIntegrador[es].pdf](ProyectoIntegrador[es].pdf).

# Archivos
<img width="321" height="527" alt="imagen" src="https://github.com/user-attachments/assets/0c1bbd42-4731-4895-89ff-40c3709bc8f3" />

Cada carpeta incluye los datos Preprocesados de un problema de clasificación. Esos datos que están en la sub-carpeta `/Bin` son introducidos en el programa `Red.exe` y ahí son procesados por redes neuronales de distintos números de capas y neuronas mediante el algoritmo de Backpropagation para la solución de cada problema de clasificación. Los resultados del entrenamiento de cada una de las redes neuronales son expuestos en la carpeta `Reg` para finalmente ser procesados por un script de python `analisis_de_registros.py` que los convierte a archivos legibles `.xlss` y gráficas `.png`.

Para obtener los datasets y preprocesarlos para su posterior entrenamiento en CUDA se usó el script "generate_cuda_datasets.py".
El archivo principal que entrena las redes neuronales es "Red.cu" y la implementación del algoritmo de Backpropagation se hizo en sobre la API de Cuda BLAS. 

<img width="1600" height="700" alt="imagen" src="https://github.com/user-attachments/assets/7efcf2aa-c0bc-4c47-bccc-3c32ed82c44e" />

Ejemplo de la comparativa de la solución del problema de Spiral por redes neuronales de 9 neuronas, pero distintas arquitecturas, una de 1 capa oculta con 9 neuronas y otra de 3 capas ocultas, cada una de 3 neuronas. Los resultados se muestran favorables para la Red Angosta pues fue capaz de resolver el problema de clasificación en cada uno de sus intentos, mientras que la red profunda solo fue capaz en uno de ellos.
