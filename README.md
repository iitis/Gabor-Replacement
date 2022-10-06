# Replacing the CNN first-layer filters with the Gabor filters

A repository with functions that can be used to replace the first-layer filters of the pretrained RGB-operating CNN networks with the Gabor or random filters. Additionally, some examples can be explored.

## A repository consists of the following files:
* **Gabor_replacement_CNNs.py** - a file with 3 functions that can be used to replace the first-layer filters of a CNN
* **Gabor_replacement_demo.ipynb** - an example of a filter replacement with the Gabor filters (chosen with Cosine Similarity)
* **Random_filter_replacement_demo.ipynb** - an example of a filter replacement with the random filters (chosen with Cosine Similarity)
* indian_elephant.jpg - an example image with an indian elephant in a zoo used in the examples

## We provide three functions that generate filter bank/replace the CNN filters:
* **generate_Gabor_filter_bank**(filter_size, period_min, period_max, deg_min, deg_max, deg_step, phase_min, phase_max, phase_step, gaussian_std_min, gaussian_std_max, gaussian_std_step) - function generates a Gabor filter bank. It is based on a cv2 function that generates Gabor filters. It takes a argumens the following parameters: filter size (the same as the first-layer filter size - if a filter has dimension 3x3 it should be 3), min, max period (in px), min, max, step for the degree, phase and std of the Gaussian envelope - see cv2 docs https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gae84c92d248183bd92fa713ce51cc3599 
* **generate_matching_random_filter_bank**(gabor_filter_bank) - it generates a random filter bank of the same size as a generated Gabor filter bank (it camn be used as a baseline for performance comparison)
* **replace_CNN_filters_with_Gabor**(model, gabor_bank, model_layer_index) - it replaces the first-layer filters (index of the first conv layer should be given as a third parameter of the function) of a model given as a parameter with the most similar filters from the given filterbank (gabor_bank) according to Cosine Similarity

We provide additional instructions on how to replicate the experiments described in our paper "On the similarity of Gabor filters and CNN first-layer filters" in file gabor_instructions.pdf.



