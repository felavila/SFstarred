from copy import deepcopy
import numpy as np

import numpy as np

def normalize_data_error(data_cutout, err_map, exptime_seconds,mask):
    data_cutout = np.atleast_2d(data_cutout).astype(float)
    err_map     = np.atleast_2d(err_map).astype(float)

    # ------------------------------------------------------------------ #
    # 1. CONVERT to e-
    # ------------------------------------------------------------------ #
    data_counts = data_cutout * exptime_seconds
    err_counts  = err_map     * exptime_seconds

    # ------------------------------------------------------------------ #
    # 2. SKY in e-
    # ------------------------------------------------------------------ #
    image_size = data_counts.shape[0]
    corner = max(1, int(0.1 * image_size))

    corners = np.concatenate([
        (data_counts*mask)[:corner,  :corner ].ravel(),
        (data_counts*mask)[:corner,  -corner:].ravel(),
        (data_counts*mask)[-corner:, :corner ].ravel(),
        (data_counts*mask)[-corner:, -corner:].ravel(),
    ])

    sky_level = np.median(corners)
    data_sky_sub = data_counts - sky_level*1     # negative values are FINE, keep them

    # ------------------------------------------------------------------ #
    # 3. SKY NOISE
    # ------------------------------------------------------------------ #
    mad = np.median(np.abs(corners - sky_level))
    sigma2_sky = (1.4826 * mad) ** 2

    # ------------------------------------------------------------------ #
    # 4. NOISE MAP — key fix: shot noise uses the *sky-included* signal
    #    because the sky photons still contribute shot noise even after
    #    subtracting the sky level. Never clip the variance term.
    # ------------------------------------------------------------------ #
    if not np.all(err_counts == 0):
        noise_map = np.sqrt(err_counts**2 + sigma2_sky)
    else:
        # shot noise: use sky-included counts as the Poisson rate
        # sky_level is always positive so this prevents negative variance
        signal_for_shot = data_sky_sub + sky_level   # add sky back ONLY for variance
        sigma2_shot = signal_for_shot.clip(min=sky_level)  # floor at sky level
        noise_map = np.sqrt(sigma2_shot + sigma2_sky)

    return data_sky_sub, noise_map          # both in e-



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