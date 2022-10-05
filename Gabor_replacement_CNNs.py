def generate_Gabor_filter_bank(filter_size, period_min, period_max, deg_min, deg_max, deg_step, phase_min, 
                               phase_max, phase_step, gaussian_std_min, gaussian_std_max, gaussian_std_step):
    """ Function allows to generate a Gabor filter bank. Such a filter bank can be used as a source of filters
    for the first-layer filters replacement for CNNs. A results of the function is a numpy array with the filters.
    """
    import cv2
    import numpy as np
    gabor_filters = []
    # We do not consider filters with a spacial aspect ratio different than one
    spacial_asect_ratio = 1
    for period in range(period_min, period_max):
        for deg in np.arange(deg_min, deg_max, deg_step):
                for phase in np.arange(phase_min, phase_max, phase_step):
                    for mean_kernel_std in np.arange(gaussian_std_min, gaussian_std_max, gaussian_std_step):
                        deg_in_radians = deg * np.pi / 180
                        gabor_filters.append(cv2.getGaborKernel((filter_size, filter_size), 
                                                                mean_kernel_std, deg_in_radians, period, 
                                                                spacial_asect_ratio, phase, ktype=cv2.CV_64F))
    return np.array(gabor_filters)



def replace_CNN_filters_with_Gabor(model, gabor_bank, model_layer_index):
    """ The following function can be used to replace the trained CNN filters (of a pretrained model) 
    of the first convolutional layer of the network operating on the RGB images
    with the most similar filters (according to cosine similarity) from a predefine Gabor filter bank
    (gabor_bank).
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
#     print('hello')
    if (gabor_bank.shape[1] != gabor_bank.shape[2]) or (gabor_bank.shape[1] != 
                                                                      model.layers[model_layer_index].get_weights()[0].shape[1]):
        print('Given model and Gabor bank have incompatible shapes')
        return 0
    
    gabor_filter_bank_for_similarity = np.reshape(gabor_bank, (gabor_bank.shape[0], gabor_bank.shape[1] ** 2))
    num_of_filters = model.layers[model_layer_index].get_weights()[0].shape[3]
    new_weights = []
    
    for i in range(0, num_of_filters):
#         print(i)
        r = model.layers[model_layer_index].get_weights()[0][:,:,0,i]
        r = np.expand_dims(np.reshape(r, (r.shape[0] * r.shape[1])), axis=0)
        gabor_r = gabor_bank[np.argmax(cosine_similarity(r, gabor_filter_bank_for_similarity))]
        gabor_r = gabor_r * np.sqrt(np.sum(r ** 2)) / np.sqrt(np.sum(gabor_r ** 2))
        
        g = model.layers[model_layer_index].get_weights()[0][:,:,1,i]
        g = np.expand_dims(np.reshape(g, (g.shape[0] * g.shape[1])), axis=0)
        gabor_g = gabor_bank[np.argmax(cosine_similarity(g, gabor_filter_bank_for_similarity))]
        gabor_g = gabor_g * np.sqrt(np.sum(g ** 2)) / np.sqrt(np.sum(gabor_g ** 2))
        
        b = model.layers[model_layer_index].get_weights()[0][:,:,2,i]
        b = np.expand_dims(np.reshape(b, (b.shape[0] * b.shape[1])), axis=0)
        gabor_b = gabor_bank[np.argmax(cosine_similarity(b, gabor_filter_bank_for_similarity))]
        gabor_b = gabor_b * np.sqrt(np.sum(b ** 2)) / np.sqrt(np.sum(gabor_b ** 2))
        
        filter_new = np.dstack([gabor_r, gabor_g, gabor_b])
        new_weights.append(filter_new)

    new_weights = np.array(new_weights)
    new_weights = np.moveaxis(new_weights, 0, -1)
    
    if len(model.layers[model_layer_index].get_weights()) == 1:
        model.layers[model_layer_index].set_weights([new_weights])
    else:
#         print(model.layers[model_layer_index].get_weights()[1])
        bias = model.layers[model_layer_index].get_weights()[1]
        model.layers[model_layer_index].set_weights([new_weights, bias])
    return model



def generate_matching_random_filter_bank(gabor_filter_bank):
    """ Function allows to generate a random filter bank of the same 
    size as a certain Gabor filter bank. 
    Such a filter bank can be used as a source of filters
    for the first-layer filters replacement for CNNs. A results of the 
    function is a numpy array with the filters.
    """
    import numpy as np
    dim0, dim1, dim2 = gabor_filter_bank.shape
    return np.random.rand(dim0, dim1, dim2)