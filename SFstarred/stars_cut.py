from copy import deepcopy
import numpy as np

def normalize_data_error(data_cutout,exp_map,star_num_pix=59,star_to_get_max=0,print_=True):
    "data_cutout =array.shape = (n,star_num_pix,star_num_pix)"
    data_cutout_copy = deepcopy(data_cutout)
    N,image_size,image_size = data_cutout.shape#len(positions_stars)
    #image_size = star_num_pix
    sigma2 = np.zeros(data_cutout.shape)
    # loop over the cutouts
    for i in range(N):
        # extract corners of the cutout, assumed to not contain any signal
        data_only_sky = data_cutout[i,int(0.9*image_size):,int(0.9*image_size):]  # NOTE: here a single one, but better to take as many corners as possible

        # estimate the standard deviation of background noise using MAD (https://en.wikipedia.org/wiki/Median_absolute_deviation)
        mad = np.median(np.abs(data_only_sky - np.median(data_only_sky)))
        sigma2_sky = (1.48 * mad)**2   # taking the square root of the standard deviation
        
        exp_map_i = np.copy(exp_map[i, :, :])
        # get the exposure map for that cutout
        exp_map_i[exp_map_i == 0.] = np.mean(exp_map_i[exp_map_i != 0.])  # just to ensure that no pixel has zero exposure time

        # shot noise variance is data / t_exp
        sigma2_target =data_cutout[i,:,:].clip(min=0) / exp_map_i  #clipping negative values because variance cannot be negative
        
        # add the two variance terms together
        #sigma2_sky[i] this should be changed because I guess it was an array
        sigma2[i,:,:] = sigma2_sky + sigma2_target
        if print_:
            print("Mean sigma2_sky =", sigma2_sky.mean())
            print("Mean sigma2_target =", sigma2_target.mean())
    sigma2_copy = deepcopy(sigma2)
    # ==> sigma2 now contain the total noise variance per pixel per star cutout
    #Renormalise your data and the noise maps by the max of the first image. Works better when using adabelief
    max_star = np.argmax([np.max(i) for i in data_cutout])
    norm = data_cutout[max_star].max() / 100. #BEWARE some stars might be brighter than others
    data_cutout /= norm
    sigma2 /= norm**2 
    return norm,data_cutout,sigma2,data_cutout_copy,sigma2_copy