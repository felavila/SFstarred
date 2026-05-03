from copy import deepcopy
import numpy as np

def normalize_data_error(data_cutout,exp_map,star_to_get_max=0,print_=False):
    "data_cutout =array.shape = (n,star_num_pix,star_num_pix)"
    data_cutout = np.atleast_3d(data_cutout)
    exp_map = np.atleast_3d(exp_map)
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

import numpy as np


def make_sigma2_from_hst_weight(
    data,
    weights,
    exptime=None,
    weight_type="exptime",
    sky_mask=None,
    sky_rms=None,
    gain=1.0,
    include_poisson=True,
    normalize_weights=True,
):
    """
    Build an approximate variance map sigma^2 from image data and a weight map.

    Parameters
    ----------
    data : ndarray
        Science image data. Usually HST drizzled images are in e-/s.

    weights : ndarray
        Weight map associated with the science image.

        If weight_type="exptime", this is interpreted as an effective exposure
        or relative exposure map, as in many HST drizzle products.

        If weight_type="ivm", this is interpreted as inverse variance.

    exptime : float, optional
        Total exposure time. Used only when weight_type="exptime" and
        normalize_weights=True.

    weight_type : {"exptime", "ivm"}, optional
        Type of weight map.

    sky_mask : ndarray of bool, optional
        Boolean mask selecting sky/background pixels used to estimate sky RMS.
        True means use this pixel.

    sky_rms : float, optional
        Background RMS in the same units as data. If not given, it is estimated
        from data[sky_mask] or all finite pixels.

    gain : float, optional
        Effective gain. For data in e-/s, gain=1 is usually fine if data
        already represents electrons per second.

    include_poisson : bool, optional
        If True, include an approximate source Poisson term.

    normalize_weights : bool, optional
        If True and weight_type="exptime", rescale weights so max(weights)
        corresponds to exptime.

    Returns
    -------
    sigma2 : ndarray
        Variance map in the same units as data squared.

    sigma : ndarray
        Standard deviation map in the same units as data.

    exptime_eff : ndarray or None
        Effective exposure-time map. Returned only for weight_type="exptime".
    """
    data = np.asarray(data, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if data.shape != weights.shape:
        raise ValueError(
            f"data and weights must have the same shape. "
            f"Got {data.shape} and {weights.shape}."
        )

    bad_weight = ~np.isfinite(weights) | (weights <= 0)

    if weight_type == "ivm":
        sigma2 = np.full_like(data, np.nan, dtype=float)
        good = ~bad_weight
        sigma2[good] = 1.0 / weights[good]
        sigma = np.sqrt(sigma2)
        return sigma2, sigma, None

    if weight_type != "exptime":
        raise ValueError("weight_type must be either 'exptime' or 'ivm'.")

    exptime_eff = weights.copy()
    exptime_eff[bad_weight] = np.nan

    if normalize_weights:
        if exptime is None:
            raise ValueError(
                "exptime must be given when normalize_weights=True."
            )

        wmax = np.nanmax(exptime_eff)

        if not np.isfinite(wmax) or wmax <= 0:
            raise ValueError("weights contain no valid positive values.")

        exptime_eff = exptime_eff / wmax * exptime

    if sky_rms is None:
        if sky_mask is None:
            sky_mask = np.isfinite(data) & np.isfinite(exptime_eff)
        else:
            sky_mask = np.asarray(sky_mask, dtype=bool)
            sky_mask = sky_mask & np.isfinite(data) & np.isfinite(exptime_eff)

        sky_rms = np.nanstd(data[sky_mask])

    # Background variance scales approximately as 1 / effective exposure time.
    exptime_ref = np.nanmax(exptime_eff)

    sigma2_bkg = sky_rms**2 * exptime_ref / exptime_eff

    if include_poisson:
        # Approximate source Poisson variance.
        # For science data in e-/s:
        #
        # counts = data * exptime_eff
        # variance_counts ≈ abs(counts)
        # variance_rate = variance_counts / exptime_eff**2
        #               = abs(data) / exptime_eff
        sigma2_poisson = np.abs(data) / (gain * exptime_eff)
        sigma2 = sigma2_bkg + sigma2_poisson
    else:
        sigma2 = sigma2_bkg

    bad = bad_weight | ~np.isfinite(data)
    sigma2[bad] = np.nan

    sigma = np.sqrt(sigma2)

    return sigma2, sigma, exptime_eff