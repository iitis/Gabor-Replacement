# Replacing the CNN first-layer filters with the Gabor filters

**NOTE:** This is a repository with methods, instructions and examples on how to use the components created for the purpose of paper **"Measuring the similarity of Gabor filters and CNN first-layer filters"**. We provide implementation of the vital software components and methods used in the study (**we plan to release our repository upon acceptance**). All of the methods are implemented  in such a way as to make them as flexible as possible not only for the easy reproduction of our experiments but hopefully also for other potential studies of other researchers. We made sure to add comments and guidelines on how to use our methods.  

We present a repository with functions that can be used to replace the first-layer filters of the pretrained RGB-operating CNN networks with the Gabor or random filters. Additionally, some examples can be explored.


While in this repository, we present the operation of our method on an example photo, all of the numerical results presented in the paper have been obtained using the following freely available datasets:
* Imagenette - see https://github.com/fastai/imagenette
* MNIST - https://keras.io/api/datasets/mnist
* Fashion MNIST - https://keras.io/api/datasets/fashion/mnist
* NIPS17 Adversarial Attacks and Defenses Competition - https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset (we used its DEV dataset for tests with the adversarial attack)


## A repository consists of the following files with code:
* **Gabor_replacement_CNNs.py** - a file with 3 functions that can be used to replace the first-layer filters of a CNN
* **Gabor_replacement_demo.ipynb** - an example of a filter replacement with the Gabor filters (chosen with Cosine Similarity)
* **Random_filter_replacement_demo.ipynb** - an example of a filter replacement with the random filters (chosen with Cosine Similarity)
* indian_elephant.jpg - an example image with an indian elephant in a zoo used in the examples

We also provide an HTML version of the notebooks.

## We provide three functions that generate filter bank/replace the CNN filters:
* **generate_Gabor_filter_bank**(filter_size, period_min, period_max, deg_min, deg_max, deg_step, phase_min, phase_max, phase_step, gaussian_std_min, gaussian_std_max, gaussian_std_step) - function generates a Gabor filter bank. It is based on a cv2 function that generates Gabor filters. It takes a argumens the following parameters: filter size (the same as the first-layer filter size - if a filter has dimension 3x3 it should be 3), min, max period (in px), min, max, step for the degree, phase and std of the Gaussian envelope - see cv2 docs https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gae84c92d248183bd92fa713ce51cc3599 
* **generate_matching_random_filter_bank**(gabor_filter_bank) - it generates a random filter bank of the same size as a generated Gabor filter bank (it camn be used as a baseline for performance comparison)
* **replace_CNN_filters_with_Gabor**(model, gabor_bank, model_layer_index) - it replaces the first-layer filters (index of the first conv layer should be given as a third parameter of the function) of a model given as a parameter with the most similar filters from the given filterbank (gabor_bank) according to Cosine Similarity

We provide additional instructions on how to replicate the experiments described in our paper "On the similarity of Gabor filters and CNN first-layer filters" in file gabor_instructions.pdf.

The following vital Python libraries have to be installed for the example to work correctly:
* NumPy - version 1.23.2
* TensorFlow (+ Keras) - version 2.10.0
* opencv-contrib-python - version 4.6.0.66
* matplotlib - version 3.4.3
* scikit-image - version 0.16.2

Additionally, for measuring accuracy to obtain the numerical results, we used SciKit-learn - version 1.1.2.




