# Basic imports
import os
import numpy as np
from copy import deepcopy
import time
import glob
from functools import partial
from pprint import pprint
import astropy
from astropy.wcs import WCS
from astropy.io import fits
from astropy import stats
from astropy.nddata import Cutout2D
import pandas as pd
import matplotlib.pyplot as plt
from astropy.visualization.wcsaxes import WCSAxes
import matplotlib.cm as cm
import pyregion
import h5py
from pyregion.mpl_helper import properties_func_default
from photutils.detection import DAOStarFinder, find_peaks
from photutils.aperture import CircularAperture
from matplotlib.colors import SymLogNorm, LogNorm, Normalize, TwoSlopeNorm
from astropy.wcs.utils import pixel_to_skycoord
from matplotlib import colors
from pathlib import Path

def find_nearest_object(array, coord):
        coord_rep = np.repeat(np.array(coord)[None, :], len(array), axis=0)
        metric = np.sqrt((array[:, 0]-coord_rep[:, 0])**2 + (array[:, 1]-coord_rep[:, 1])**2)
        idx = np.argmin(metric, axis=0)
        return array[idx]
def create_rectangle_patch(data,center, size=1.0):
    x, y = center
    width = height = size
    d = data[int(y - height / 2): int(y + height / 2),int(x - width / 2):int(x + width / 2)].copy()
    if np.all(np.isnan(d)):
          #print(np.nanmax(d))
          return plt.Rectangle((x - width / 2, y - height / 2), width, height, edgecolor='k', facecolor='none'), np.zeros((0,0)),[x, y]
    return plt.Rectangle((x - width / 2, y - height / 2), width, height, edgecolor='red', facecolor='none'), d,[x, y]#data[int(x_m - width / 2):int(x_m + width / 2), int(y_m - height / 2): int(y_m + height / 2)]
###########To get mag#########################
def get_stmag(flux, photflam, photzpt):
   flux = np.asarray(flux)
   scalar_input = False
   if flux.ndim == 0:
      flux = flux[None] 
      scalar_input = True
   flux = flux * photflam
   mag = -2.5 * np.log10(flux) + photzpt

   if scalar_input:
       return np.squeeze(mag)

   return mag


def get_abmag(flux, photflam, photzpt, photplam):
   stmag = get_stmag(flux, photflam, photzpt)

   return stmag - 5. * np.log10(photplam) + 2.5 * np.log10(299792458e10) - 27.5
#header keys values from my images from HST
header_values ={'F814W': {'PHOTFLAM': 1.516962e-19,
  'PHOTZPT': -21.1,
  'PHOTPLAM': 8032.93505,
  'PHOTFNU': 3.2921571625e-07,
  'PHOTBW': 665.299255},
 'F475X': {'PHOTFLAM': 1.561348275e-19,
  'PHOTZPT': -21.1,
  'PHOTPLAM': 4940.7244,
  'PHOTFNU': 1.2787481875e-07,
  'PHOTBW': 660.4336249999999},
 'F160W': {'PHOTFLAM': 1.9429001e-20,
  'PHOTZPT': -21.1,
  'PHOTPLAM': 15369.176,
  'PHOTFNU': 1.5308434e-07,
  'PHOTBW': 826.25085}}


def normalize_data_error(data_cutout,exp_map,star_num_pix=59,star_to_get_max=0,print_=True):
    "data_cutout =array.shape = (n,star_num_pix,star_num_pix)"
    data_cutout_copy = deepcopy(data_cutout)
    N = data_cutout.shape[0]#len(positions_stars)
    image_size = star_num_pix
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
        sigma2_target =data_cutout[i,:,:].clip(min=0) / exp_map_i  # clipping negative values because variance cannot be negative
        

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

def data_for_starred(name_file,star_num_pix=61,where_is_data='HST_IMAGES_0214_2105' \
                     ,star_to_get_max=0,reg_name="reg_without_system_more.reg"\
                        ,remove_star = [],fwhm_source=2,threshold_source=5,save=True,shift = [0,0],make_plots=False):
    """Remove_star is a list that can contain numbers from 1 to n. 
    If you want to remove a star from the region, add the number. 
    It will be kept in the complete image for visualization purposes.
    #for lenstronomy star_num_pix should be odd
    fwhm_source and threshold_source are parameters to find the point source image also necesary for starred"""
    if os.path.isdir("data_products")==False:
        os.mkdir("data_products")
    hst_image = os.path.join(where_is_data,name_file)
    data_raw, header = fits.getdata(hst_image, header=True)
    wcs = WCS(header)
    data_raw = data_raw.astype(float)
    #Get_stats_properties
    mean, median, std = stats.sigma_clipped_stats(data_raw, sigma=5.0)
    print(f"Statistical properties data raw \n mean={mean}\n median={median}\n std={std}")
    #plots parameters
    vmin,vmax = np.nanmin(data_raw) * 0.5, np.nanmax(data_raw) * 0.1
    #
    data_raw = data_raw - median
    #parameters selected by hand
    drc,a,detec_fwhm,detec_threshold = "drc",0.5,5,10.*std
    if "F160W" in hst_image:
        drc,FILTER,a,detec_fwhm,detec_threshold = "drz", "F160W",1,5.,50.*std
    elif 'F814W' in hst_image:
        FILTER = 'F814W'
    elif "F475X" in hst_image:
        FILTER = "F475X"
    else:
        raise NotImplementedError
    path_filter = os.path.join("data_products",FILTER)
    if os.path.isdir(path_filter)==False:
        os.mkdir(path_filter)
    num_detectors = 1
    exp_time_hdr = float(header["EXPTIME"]) / float(num_detectors) # "EXPTIME" exposure duration (seconds)--calculated
    print(f"Using reference exposure time of {exp_time_hdr} s")
    #Weight maps data set up
    wht_path = os.path.join(where_is_data, f'WG0214-2105_{FILTER}_{drc}_wht.fits')
    weights,_ = fits.getdata(wht_path, header=True)
    weights = weights.astype(float)
    #In my case the external parts are 0 so I changed those values to NaN because otherwise they turn the median to 0 
    RE_NORMALIZE_WEIGHTS = True
    weights[weights==0] = np.NaN
    wht_mean = weights[weights>0].mean()
    wht_max = weights[weights>0].max()
    wht_std = weights[weights>0].std()
    print("Stats wht (mean, std, max):", wht_mean, wht_std, wht_max)
    if RE_NORMALIZE_WEIGHTS:
        weights = weights / wht_max * exp_time_hdr
        print("Re-normalized the WHT map.")
        print("Median afterwards:", np.nanmedian(weights))
    #cut_out_regions
    reg_path= os.path.join(where_is_data,reg_name)
    reg_to_sky_frame = pyregion.open(reg_path).as_imagecoord(header=header)
    cut_out_list = []
    star_pos_list = []
    for reg in reg_to_sky_frame:
        center_reg = reg.coord_list[:2]
        size = 40.0
        rectangle_patch,cut_out,star_pos = create_rectangle_patch(data_raw,center_reg, size=size)
        if cut_out.shape != (size,size):
            continue
        cut_out_list.append(cut_out)
        star_pos_list.append(star_pos)
    #############Look for points sources###############
    daofind = DAOStarFinder(fwhm=detec_fwhm, threshold=detec_threshold)  
    sources = daofind(data_raw)
    positions = np.transpose((sources['xcentroid'], sources['ycentroid'])) #inverted
    apertures_daostars = CircularAperture(positions, r=5.0)
    ###########Search for the nearest to our images###################
    positions_stars = np.stack([find_nearest_object(positions, star_pos) for star_pos in star_pos_list])
    apertures_stars = CircularAperture(positions_stars, r=10.0)
    ##############################
    if make_plots:
        plt.figure(figsize=(20, 20))
        plt.imshow(data_raw, cmap='Greys', norm=LogNorm(1e-6))
        apertures_daostars.plot(color='blue', lw=1.5, alpha=0.5)
        apertures_stars.plot(color='red', lw=1.5, alpha=0.5)
        for i, star_pos in enumerate(star_pos_list):
            plt.text(star_pos[0], star_pos[1], i+1, color='red')
        if save:
            plt.savefig(os.path.join(path_filter,f"fits_with_stars_{FILTER}.pdf"))
        plt.show()
    cutouts_stars = [
        Cutout2D(data_raw, positions_stars[i], star_num_pix, wcs=wcs)
        for i in range(len(positions_stars))
    ]
    cutouts_weights = [
        Cutout2D(weights, positions_stars[i], star_num_pix, wcs=wcs)
        for i in range(len(positions_stars))
    ]
    ##############system handling##################################
    cutout_system = Cutout2D(data_raw,wcs.all_world2pix(header["RA_TARG"]+shift[0], header["DEC_TARG"]+shift[1], 0), star_num_pix, wcs=wcs)
    cutout_system_weight = Cutout2D(weights,wcs.all_world2pix(header["RA_TARG"]+shift[0], header["DEC_TARG"]+shift[1], 0), star_num_pix, wcs=wcs)
    daofind_point_source = DAOStarFinder(fwhm=fwhm_source, threshold=threshold_source)  
    point_sources = daofind_point_source(cutout_system.data)
    positions_source = np.transpose((point_sources['xcentroid'], point_sources['ycentroid'])) #inverted
    point_sources_aperture = CircularAperture(positions_source, r=5.0)
    if make_plots:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(cutout_system.data, cmap='Greys', norm=LogNorm(1e-6))
        for i, image_pos in enumerate(positions_source):
            axes[0].text(image_pos[0], image_pos[1], i, color='red')
            point_sources_aperture.plot(color='blue', lw=1.5, alpha=0.5,ax=axes[0]);
        axes[1].imshow(cutout_system_weight.data)
        axes[0].set_title(f"Cut out system {star_num_pix}x{star_num_pix}pix centered  in ra:{header["RA_TARG"]}°,dec:{header["DEC_TARG"]}°")
        axes[1].set_title("Exposure map")
        if save:
            plt.savefig(os.path.join(path_filter,f"cut_out_system_{FILTER}.pdf"))
        plt.show()
    ########################################
    coord_list = []
    header_list = []
    for i in range(len(positions_stars)):
        if i+1 not in remove_star:
           
            ############################
            #cter = cutouts_stars[i].origin_original
            cter = positions_stars[i]
            ############################
            sky_coord = pixel_to_skycoord(*cter, wcs=wcs)
            print(sky_coord.to_string(style="hmsdms", precision=2))
            coord_list.append(sky_coord.to_string(style="hmsdms", precision=2))
            header_list.append(cutouts_stars[i].wcs.to_header())
            if make_plots:
                fig, axes = plt.subplots(1, 2, figsize=(8, 3))
                ax = axes[0]
                ax.set_title(f"Star cutout {i+1}")
                im = ax.imshow(cutouts_stars[i].data, norm=LogNorm(1e-3, 1e4), cmap='gray_r')
                fig.colorbar(im, ax=ax)
                ax = axes[1]
                ax.set_title("Exposure map")
                im = plt.imshow(cutouts_weights[i].data)
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                if save:
                    plt.savefig(os.path.join(path_filter,f"star_cutout_{i+1}_{sky_coord.to_string(style='hmsdms', precision=2)}_{FILTER}.jpg"))
                plt.show()
    data_cutout = np.stack([cutouts_stars[i].data for i in range(len(positions_stars)) if i+1 not in remove_star])
    exp_map = np.stack([cutouts_weights[i].data for i in range(len(positions_stars)) if i+1 not in remove_star])
    #norm,data_cutout,sigma2,data_cutout_copy,sigma2_copy
    norm,data_norm,sigma2_norm,data_cutout,sigma2=normalize_data_error(data_cutout,exp_map,star_num_pix=star_num_pix,star_to_get_max=star_to_get_max)
    #data_cutout is already normalized; this can be changed in a future, i.e. data_cutout == "data_cutout_norm"
    result = {"filter":f"{FILTER}","data_cutout_norm":data_norm \
              ,"sigma2_norm":sigma2_norm,"norm":norm \
              , "data_cutout":data_cutout, "sigma2":sigma2
              ,"exp_map":exp_map,"sky_coord":coord_list,"wcs_to_header":header_list \
               ,"system_cut_out":cutout_system.data,"weights_system_cut_out":cutout_system_weight.data,"image_positions":positions_source,"stats_raw_image":{"median":median,"std":std,"max":max}}
    local_psf_filename = None
    if save:
        cut_filename = f'cut_out_{FILTER}.hdf5'
        local_psf_filename = os.path.join(path_filter,cut_filename)
        f = h5py.File(local_psf_filename, "w")
        dset = f.create_dataset("data_cutout_norm", data=data_norm)
        dset = f.create_dataset("sigma2_norm", data=sigma2_norm)
        dset = f.create_dataset("norm", data=norm)
        dset = f.create_dataset("data_cutout", data=data_cutout)
        dset = f.create_dataset("sigma2", data=sigma2)
        dset = f.create_dataset("exp_map", data=exp_map)
        dset = f.create_dataset("sky_coord", data=coord_list)
        dset = f.create_dataset("system_cut_out", data=cutout_system.data)
        dset = f.create_dataset("weights_system_cut_out", data=cutout_system_weight.data)
        dset = f.create_dataset("image_positions", data=positions_source)
        f.close()
        print(f"\n Your results are save in {os.path.join(path_filter,cut_filename)} \n")
    #maybe here I should explain what I save, also cut out the system could be a good idea
    return result,local_psf_filename

def read_hdf5_to_dict(file_path):
    if file_path.endswith(".hdf5") == False:
        return print("maybe the file is not a h5py")
    data_dict = {}
    filename =file_path
    with h5py.File(filename, 'r') as f:
        # Recursively traverse the HDF5 file
        def traverse(item, path=''):
            if isinstance(item, h5py.Dataset):
                data_dict[path] = item[()]
            elif isinstance(item, h5py.Group):
                for key in item.keys():
                    traverse(item[key], path+ key)
        traverse(f)
    return data_dict
def save_dict_as_h5py(my_dict,file_path):
    if file_path.endswith(".hdf5") == False:
        file_path = file_path + ".hdf5"
    # Open the HDF5 file in write mode
    with h5py.File(file_path, 'w') as f:
        # Iterate over the dictionary items and save each one as a dataset
        for key, value in my_dict.items():
            f.create_dataset(key, data=value)