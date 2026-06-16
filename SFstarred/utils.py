
import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from multiprocessing import Pool, cpu_count
import os 
import pandas as pd 
from pathlib import Path
import pickle 
from scipy.ndimage import shift
from photutils.centroids import centroid_com
from astropy.stats import sigma_clipped_stats
import sep
from scipy.ndimage import binary_dilation

from SFstarred.plots import plot_cutouts

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

def save_pickle(starred_dict,save_path,verbose=True):
	save_path = Path(save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)
	with open(save_path, "wb") as f:
		pickle.dump(starred_dict, f)
	if verbose:
		print(f"Saved dictionary to: {save_path}")
  
def save_npy(narrow_psfs,save_path,verbose=True):
	save_path = Path(save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(save_path, narrow_psfs)
	if verbose:
		print(f"Saved npy to: {save_path}")
  
def _read_one_fits(args):
    full_path, file, keys, add_4most_keys, keys_4most = args

    try:
        primary_header = fits.getheader(full_path, ext=0)
        row = [full_path, file]
        row.extend(primary_header.get(key, None) for key in keys)

        if add_4most_keys:
            try:
                ext1_header = fits.getheader(full_path, ext=1)
                row.extend(ext1_header.get(key, None) for key in keys_4most)
            except Exception:
                row.extend([None] * len(keys_4most))

        return row
    except Exception:
        return None



def make_a_fits_list_csv(path, save=False, add_eso_keys=False, add_4most_keys=False,add_keys=[]):
    base_keys = ["TARGPROP","TARGNAME", "TARG_RA", "TARG_DEC", "RA_TARG","DEC_TARG","TELESCOP", "INSTRUME", "DATE","EXPTIME","FILTER"] + add_keys

    eso_keys = [
        "HIERARCH ESO DPR CATG",
        "HIERARCH ESO ADA POSANG",
        "HIERARCH ESO ADA ABSROT END",
        "HIERARCH ESO ADA ABSROT START",
        "HIERARCH ESO TEL TARG ALPHA",
        "HIERARCH ESO TEL TARG DELTA",
        "HIERARCH ESO OBS NAME",
        "HIERARCH ESO OBS PROG ID",
    ]

    keys_4most = ["SRVID1", "SRVID2", "SRVID3","SPECUID"]
    keys = list(dict.fromkeys(base_keys + (eso_keys if add_eso_keys else [])))
    
    if isinstance(path,str):
        tasks = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".fits") and "c1" not in file and "c2" not in file:
                    full_path = os.path.join(root, file)
                    tasks.append((full_path, file, keys, add_4most_keys, keys_4most))
    else:
        tasks = []
        for full_path in path:
            if full_path.endswith(".fits") and "c1" not in full_path and "c2" not in full_path:
                tasks.append((full_path, full_path.split("/")[-1], keys, add_4most_keys, keys_4most))

    with Pool(processes=cpu_count()) as pool:
        rows = pool.map(_read_one_fits, tasks)

    rows = [r for r in rows if r is not None]

    final_keys = keys + (keys_4most if add_4most_keys else [])
    df = pd.DataFrame(rows, columns=["path", "file_name", *final_keys])

    if "DATE" in df.columns:
        df = df.sort_values("DATE", kind="stable")

    if save:
        df.to_csv(save, index=False)

    return df

def make_cutouts(
    image,
    sigma2,
    star_ids,
    xis,
    yis,
    rpix,
    scale_stars=True,
    sub_pixel=True,
    show_figs=True,
    verbose=True,
):
    """
    Extract postage-stamp cutouts from an image and its variance map.

    Parameters
    ----------
    image : np.ndarray
        Science image.
    sigma2 : np.ndarray
        Variance image, i.e. sigma**2 for each pixel.
    star_ids : list
        Source IDs.
    xis : list
        x-centroids.
    yis : list
        y-centroids.
    rpix : int
        Half-size of the cutout.
    scale_stars : bool, default=True
        If True, divide each stellar cutout by its peak flux. The variance
        cutout is divided by peak_flux**2.
    sub_pixel : bool, default=True
        If True, align each cutout at the sub-pixel level.
    show_figs : bool, default=True
        If True, show diagnostic figures.
    verbose : bool, default=True
        If True, print diagnostic information.

    Returns
    -------
    image_list : list
        List of image cutouts.
    sigma2_list : list
        List of variance cutouts.
    """

    print(f"\nCalling make_cutouts with {len(star_ids)} sources.\n")

    image_list = []
    sigma2_list = []
    new_centers = []
    for i in range(len(xis)):

        xi = int(np.rint(xis[i]))
        yi = int(np.rint(yis[i]))
        star_id = star_ids[i]

        print(f"Star ID {star_id}: (x,y) = ({xi}, {yi})")

        if verbose:
            print("The read in x, y are:", xis[i], yis[i])

        subimage = image[yi - rpix - 1 : yi + rpix, xi - rpix - 1 : xi + rpix]
        subsigma2 = sigma2[yi - rpix - 1 : yi + rpix, xi - rpix - 1 : xi + rpix]

        if sub_pixel:
            x_shift = round(xi - xis[i], 5)
            y_shift = round(yi - yis[i], 5)

            if verbose:
                print("x_shift, y_shift =", x_shift, y_shift)

            mask = np.array(
                [
                    [True, False, True],
                    [False, False, False],
                    [True, False, True],
                ]
            )

            xcom, ycom = centroid_com(
                subimage[rpix - 1 : rpix + 2, rpix - 1 : rpix + 2],
                mask=mask,
            )

            if verbose:
                print(
                    "Before shift centroid =",
                    round(xcom + rpix - 1, 4),
                    round(ycom + rpix - 1, 4),
                )

            my_shift = [y_shift, x_shift]

            subimage = shift(subimage, my_shift, mode="mirror")

            # Since this is variance sigma^2, keep it positive.
            # Interpolating variance directly is okay for alignment,
            # but avoid creating negative values from interpolation.
            subsigma2 = shift(subsigma2, my_shift, mode="mirror")
            subsigma2 = np.clip(subsigma2, 0.0, None)

            xcom, ycom = centroid_com(subimage[rpix - 1 : rpix + 2, rpix - 1 : rpix + 2],mask=mask,)
            new_centers.append([round(xcom + rpix - 1, 4),round(ycom + rpix - 1, 4),])
            if verbose:
                print(
                    "After shift centroid = ",
                    round(xcom + rpix - 1, 4),
                    round(ycom + rpix - 1, 4),
                )

        else:
            if verbose:
                print("Warning: Not aligning at the sub-pixel level. Is this intended?")

        peak_location = np.unravel_index(np.argmax(subimage, axis=None), subimage.shape)

        if verbose:
            print(
                f"The subimage peak flux (x,y) = "
                f"({peak_location[1]}), {peak_location[0]})"
            )

        peak_flux = np.nanmax(subimage)

        if scale_stars:
            print("Scaling the stars peak flux to unity...")

            if not np.isfinite(peak_flux) or peak_flux == 0:
                print("Invalid peak flux. This object will be excluded.\n")
                continue

            subimage = subimage / peak_flux
            subsigma2 = subsigma2 / peak_flux**2

        if peak_location[1] != 0 and peak_location[0] != 0:
            image_list.append(subimage)
            sigma2_list.append(subsigma2)

            if show_figs:
                plot_cutouts(data=subimage, rpix=rpix)

        else:
            print("This object does not have a central peak and will be excluded.\n")

    return image_list, sigma2_list, np.array(new_centers)





import numpy as np
import sep
from scipy.ndimage import binary_dilation

def mask_surrounding_stars(
    data,
    noisemap,
    thresh=3.0,
    minarea=2,
    dilation_radius=5,
    deblend_cont=0.005,
    manual_masks=None,
    central_protection_radius=3.0,
    central_saturated_radius=None,
):
    """
    Mask detected neighbouring sources plus optional manually defined close sources.

    Pixels with signal < 0 near the image center can also be masked, useful for
    saturated/bad central pixels.

    Returns
    -------
    mask : 2D bool array
        True = good pixel, False = masked.

    n_masked : int
        Number of automatically masked SEP neighbour detections.
    """

    data = np.asarray(data, dtype=float)
    noisemap = np.asarray(noisemap, dtype=float)

    data_sep = np.ascontiguousarray(data)
    err_sep = np.ascontiguousarray(noisemap)

    objects, seg_map = sep.extract(
        data_sep,
        thresh=thresh,
        err=err_sep,
        minarea=minarea,
        segmentation_map=True,
        deblend_cont=deblend_cont,
    )

    mask = np.ones_like(data, dtype=bool)

    ny, nx = data.shape
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0

    n_masked = 0

    if len(objects) > 0:
        distances = np.sqrt((objects["x"] - cx)**2 + (objects["y"] - cy)**2)
        central_idx = np.argmin(distances)

        r = dilation_radius
        y_k, x_k = np.ogrid[-r:r + 1, -r:r + 1]
        dilation_kernel = (x_k**2 + y_k**2) <= r**2

        for i, obj in enumerate(objects):
            if i == central_idx:
                continue

            dist_to_center = np.sqrt((obj["x"] - cx)**2 + (obj["y"] - cy)**2)
            if dist_to_center < central_protection_radius:
                continue

            obj_footprint = seg_map == (i + 1)
            obj_footprint_dilated = binary_dilation(
                obj_footprint,
                structure=dilation_kernel,
            )

            mask[obj_footprint_dilated] = False
            n_masked += 1

    # Manual masks for close contaminants
    if manual_masks is not None:
        yy, xx = np.indices(data.shape)

        for x0, y0, radius in manual_masks:
            manual_region = (xx - x0)**2 + (yy - y0)**2 <= radius**2
            mask[manual_region] = False

    # Mask saturated / bad pixels near the center with negative signal
    if central_saturated_radius is not None:
        yy, xx = np.indices(data.shape)

        r_center = np.sqrt((xx - cx)**2 + (yy - cy)**2)

        central_saturated = ((r_center <= central_saturated_radius) & ~np.isfinite(np.log10(data/noisemap)) )

        mask[central_saturated] = False

    return mask, n_masked




def correct_cutout_and_noise(
    cutout,
    noisemap_slice,
    exptime_seconds,
    sky_sigma=3.0,
    sky_maxiters=10,
    source_mask_radius_frac=0.33,
    bad_error_value=1e10,
):
    """
    Given a raw image cutout and its noisemap slice from the full image,
    return the sky-subtracted image and corrected noisemap.

    Bad or negative-noise pixels are assigned a very large error instead
    of being set to NaN.
    """

    cutout = np.array(cutout, dtype=float)
    noisemap_slice = np.array(noisemap_slice, dtype=float)

    ny, nx = cutout.shape

    # Use corner boxes for sky estimation
    corner_frac = 0.20
    dy = int(ny * corner_frac)
    dx = int(nx * corner_frac)

    corner_mask = np.zeros_like(cutout, dtype=bool)

    # True means: use this pixel for sky
    corner_mask[:dy, :dx] = True
    corner_mask[:dy, -dx:] = True
    corner_mask[-dy:, :dx] = True
    corner_mask[-dy:, -dx:] = True

    # Bad pixels for sky estimation
    bad_data = ~np.isfinite(cutout)
    bad_noise = (
        ~np.isfinite(noisemap_slice)
        | (noisemap_slice <= 0)
    )

    sky_mask = ~corner_mask | bad_data | bad_noise

    _, sky_median, sky_std = sigma_clipped_stats(
        cutout,
        mask=sky_mask,
        sigma=sky_sigma,
        maxiters=sky_maxiters,
    )

    data_skysub = cutout - sky_median

    # Replace bad data values with something finite
    # They will be ignored because their error is huge.
    data_skysub = np.where(np.isfinite(data_skysub), data_skysub, 0.0)

    # Replace bad / negative / zero noise with huge error
    noisemap_clean = np.where(
        bad_noise,
        bad_error_value,
        noisemap_slice,
    )

    var = noisemap_clean**2

    # Add Poisson variance only where data is valid
    source_rate = np.clip(data_skysub, 0.0, None)
    poisson_var = source_rate / exptime_seconds

    var = var + poisson_var

    noise_corrected = np.sqrt(var)

    # Force all originally bad pixels to huge error
    bad = bad_data | bad_noise
    noise_corrected[bad] = bad_error_value

    return data_skysub, noise_corrected

def profile_psf(psf):
    if len(psf.shape)==2:
        psf = psf[None,:]
    # Radial profile of your PSF
    #psf = narrow_psfs[0]  # or whichever band
    cy, cx = np.array(psf.shape) // 2
    y, x = np.indices(psf.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    radial_profile = np.bincount(r.ravel(), psf.ravel()) / np.bincount(r.ravel())
    plt.semilogy(radial_profile)
    plt.xlabel("Radius [px]"); plt.ylabel("PSF flux"); plt.title("PSF radial profile")
    plt.show()

from copy import deepcopy
from starred.psf.psf import PSF
from starred.psf.loss import Loss, Prior
from starred.optim.optimization import Optimizer
from starred.psf.parameters import ParametersPSF
from starred.utils.noise_utils import propagate_noise
from starred.utils.generic_utils import fourier_division, gaussian_function, fwhm2sigma, make_grid, Downsample
from starred.plots import plot_function as pltf
from starred.procedures.psf_routines import sanitize_inputs

def build_psf_edit(image, noisemap, subsampling_factor,
              masks=None,
              n_iter_analytic=40, n_iter_adabelief=2000,
              guess_method_star_position='barycenter', guess_fwhm_pixels=3.,
              field_distortion=False, stamp_coordinates=None, adjust_sky=False,
              regularization_strength_scales=1,regularization_strength_hf=1,
              regularization_strength_positivity=0):
    """

    Routine taking in cutouts of stars (shape (N, nx, ny), with N the number of star cutouts, and nx,ny the shape of each cutout)
    and their noisemaps (same shape), producing a narrow PSF with pixel grid of the given subsampling_factor

    Parameters
    ----------
    image : array, shape (imageno, nx, ny)
        array containing the data
    noisemap : array, shape (imageno, nx, ny)
        array containing the noisemaps.
    subsampling_factor : int
        by how much we supersample the PSF pixel grid compare to data.
    masks: optional, array of same shape as image and noisemap containing 1 for pixels to be used, 0 for pixels to be ignored.
    n_iter_analytic: int, optional, number of iterations for fitting the moffat in the first step
    n_iter_adabelief: int, optional, number of iterations for fitting the background in the second step
    guess_method_star_position: str, optional, one of 'barycenter', 'max' or 'center'
    guess_fwhm_pixels: float, the estimated FWHM of the PSF, is used to initialize the moffat. Default 3.
    field_distortion: whether we allow the psf to vary across the field. If yes, 'stamp_coordinates' must be supplied.
    stamp_coordinates: array of shape (imageno, 2), the pixel coordinates of the different stars in data.
    adjust_sky: bool, optional, if True, the sky level is adjusted for each PSF star. Default False.

    Returns
    -------
    result : dictionary.
        dictionary containing the narrow PSF (key narrow_psf) and other useful things.

    """
    # checks
    if field_distortion and (stamp_coordinates is None):
        raise RuntimeError(
            "starred.psf_routines.build_psf: asked to include field distortions,"
            "but no star positions on the ccd (argument stamp_coordinates) provided."
        )

    # sanitize inputs: mask NaN values
    image, noisemap, masks = sanitize_inputs(image, noisemap, masks)

    # normalize by max of data(numerical precision best with scale ~ 1)
    norm = np.nanpercentile(image, 99.)
    image /= norm
    noisemap /= norm

    model = PSF(image_size=image[0].shape[0], number_of_sources=len(image),
                upsampling_factor=subsampling_factor,
                convolution_method='fft',
                include_moffat=True,
                elliptical_moffat=True,
                field_distortion=field_distortion)

    smartguess = lambda im: model.smart_guess(im, fixed_background=True, guess_method=guess_method_star_position,
                                              masks=masks, guess_fwhm_pixels=guess_fwhm_pixels, adjust_sky=adjust_sky)

    # Parameter initialization.
    kwargs_init, kwargs_fixed, kwargs_up, kwargs_down = smartguess(image)

    # smartguess doesn't know about cosmics, other stars ...
    # so we'll be a bit careful.
    medx0 = np.median(kwargs_init['kwargs_gaussian']['x0'])
    medy0 = np.median(kwargs_init['kwargs_gaussian']['y0'])
    kwargs_init['kwargs_gaussian']['x0'] = medx0 * np.ones_like(kwargs_init['kwargs_gaussian']['x0'])
    kwargs_init['kwargs_gaussian']['y0'] = medy0 * np.ones_like(kwargs_init['kwargs_gaussian']['y0'])

    parameters = ParametersPSF(kwargs_init,
                               kwargs_fixed,
                               kwargs_up=kwargs_up,
                               kwargs_down=kwargs_down)

    loss = Loss(image, model, parameters, noisemap**2, len(image),
                regularization_terms='l1_starlet',
                regularization_strength_scales=0,
                regularization_strength_hf=0,
                masks=masks,
                star_positions=stamp_coordinates)

    optim = Optimizer(loss,
                      parameters,
                      method='l-bfgs-b')

    # fit the moffat:
    best_fit, logL_best_fit, extra_fields, runtime = optim.minimize(maxiter=n_iter_analytic,
                                                                    restart_from_init=True)

    kwargs_partial = parameters.args2kwargs(best_fit)

    # now moving on to the background.
    # Release background and distortion, fix the moffat
    kwargs_fixed = {
        'kwargs_moffat': {'fwhm_x': kwargs_partial['kwargs_moffat']['fwhm_x'],
                          'fwhm_y': kwargs_partial['kwargs_moffat']['fwhm_y'],
                          'phi': kwargs_partial['kwargs_moffat']['phi'],
                          'beta': kwargs_partial['kwargs_moffat']['beta'],
                          'C': kwargs_partial['kwargs_moffat']['C']},
        'kwargs_gaussian': {},
        'kwargs_background': {},
        'kwargs_distortion': deepcopy(kwargs_init['kwargs_distortion'])
    }

    if not adjust_sky:
        kwargs_fixed['kwargs_background']['mean'] = deepcopy(kwargs_init['kwargs_background']['mean'])

    parametersfull = ParametersPSF(kwargs_partial,
                                   kwargs_fixed,
                                   kwargs_up,
                                   kwargs_down)

    # median of noisemaps, but still fully mask any pixel that might be crazy in any of the frames.
    average_noisemap = np.nanmedian(noisemap, axis=0)
    average_noisemap = np.expand_dims(average_noisemap, (0,))
    mask = np.min(masks, axis=0)
    mask = np.expand_dims(mask, (0,))
    W = propagate_noise(model=model, noise_maps=average_noisemap, kwargs=kwargs_init,
                        masks=mask,
                        wavelet_type_list=['starlet'],
                        method='MC', num_samples=100,
                        seed=1, likelihood_type='chi2',
                        verbose=False,
                        upsampling_factor=subsampling_factor)[0]

    lossfull = Loss(image, model, parametersfull,
                    noisemap**2, len(image),
                    regularization_terms='l1_starlet',
                    regularization_strength_scales=regularization_strength_scales,
                    regularization_strength_hf=regularization_strength_hf,
                    regularization_strength_positivity=regularization_strength_positivity,
                    W=W,
                    regularize_full_psf=False,
                    masks=masks,
                    star_positions=stamp_coordinates)

    optimfull = Optimizer(lossfull, parametersfull, method='adabelief')

    optimiser_optax_option = {
                                'max_iterations': n_iter_adabelief, 'min_iterations': None,
                                'init_learning_rate': 1e-4, 'schedule_learning_rate': True,
                                # important: restart_from_init True
                                'restart_from_init': True, 'stop_at_loss_increase': False,
                                'progress_bar': True, 'return_param_history': True
                              }

    best_fit, logL_best_fit, extra_fields2, runtime = optimfull.minimize(**optimiser_optax_option)

    kwargs_partial2 = parametersfull.args2kwargs(best_fit)

    # this last step is just permitting distortion in the field.
    if field_distortion:
        kwargs_fixed = {
            'kwargs_moffat': {'fwhm_x': kwargs_partial2['kwargs_moffat']['fwhm_x'],
                              'fwhm_y': kwargs_partial2['kwargs_moffat']['fwhm_y'],
                              'phi': kwargs_partial2['kwargs_moffat']['phi'],
                              'beta': kwargs_partial2['kwargs_moffat']['beta'],
                              'C': kwargs_partial2['kwargs_moffat']['C']},
            'kwargs_gaussian': {},
            'kwargs_background': {},
            'kwargs_distortion': {}
        }

        if not adjust_sky:
            kwargs_fixed['kwargs_background']['mean'] = deepcopy(kwargs_init['kwargs_background']['mean'])

        parametersfull = ParametersPSF(kwargs_partial2,
                                       kwargs_fixed,
                                       kwargs_up,
                                       kwargs_down)

        lossfull = Loss(image, model, parametersfull,
                        noisemap ** 2, len(image),
                        regularization_terms='l1_starlet',
                        regularization_strength_scales=regularization_strength_scales,
                        regularization_strength_hf=regularization_strength_hf,
                        regularization_strength_positivity=regularization_strength_positivity,
                        W=W,
                        regularize_full_psf=False,
                        masks=masks,
                        star_positions=stamp_coordinates)

        optimfull = Optimizer(lossfull, parametersfull, method='adabelief')

        optimiser_optax_option = {
            'max_iterations': 1000, 'min_iterations': None,
            'init_learning_rate': 3e-5, 'schedule_learning_rate': True,
            'restart_from_init': True, 'stop_at_loss_increase': False,
            'progress_bar': True, 'return_param_history': True
        }

        best_fit, logL_best_fit, extra_fields3, runtime = optimfull.minimize(**optimiser_optax_option)
        extra_fields2['loss_history'] = np.append(extra_fields2['loss_history'], extra_fields3['loss_history'])
        kwargs_final = parametersfull.args2kwargs(best_fit)
    else:
        kwargs_final = kwargs_partial2

    ###########################################################################
    # book keeping
    narrowpsf = model.get_narrow_psf(**kwargs_final, norm=True)
    fullpsf = model.get_full_psf(**kwargs_final, norm=True)
    numpsf = model.get_background(kwargs_final['kwargs_background'])
    moffat = model.get_moffat(kwargs_final['kwargs_moffat'], norm=True)
    fullmodel = model.model(**kwargs_final, positions=stamp_coordinates)
    residuals = image - fullmodel
    # approximate chi2: hard to count params with regularization - indicative.
    chi2 = masks * residuals**2 / noisemap**2
    valid_pixels_count = np.sum(masks)
    red_chi2 = np.sum(chi2) / valid_pixels_count
    valid_pixels_count_per_slice = np.sum(masks, axis=(1,2)) + 1e-3 
    # above, + 1e-3 to avoid potential divisions by 0.  the user should not
    # input fully masked stamps anyways, but never know.
    # if fully masked, will yield a very large reduced chi2, which would make
    # it easy to flag.
    red_chi2_per_stamp = np.sum(chi2, axis=(1, 2)) / valid_pixels_count_per_slice[:, None]

    result = {
        'model_instance': model,
        'kwargs_psf': kwargs_final,
        'narrow_psf': narrowpsf,
        'full_psf': fullpsf,
        'numerical_psf': numpsf,
        'moffat': moffat,
        'models': fullmodel,
        'residuals': residuals,
        'analytical_optimizer_extra_fields': extra_fields,
        'adabelief_extra_fields': extra_fields2,
        'chi2': red_chi2,
        'chi2_per_stamp': red_chi2_per_stamp
    }
    ###########################################################################
    return result
