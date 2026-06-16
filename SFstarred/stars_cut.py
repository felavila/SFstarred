from copy import deepcopy
import numpy as np


def normalize_data_error(
    data_cutout,
    err_map,
    exptime_seconds=None,
    mask=None,
    corner=0.1,
    input_in_electrons=True,
    verbose=True,
):
    """
    Sky-subtract a single star cutout and return data + noise map in electrons.

    Parameters
    ----------
    data_cutout : 2D array
        Image cutout. By default assumed already in e-.
    err_map : 2D array
        Error/noise map. By default assumed already in e-.
    exptime_seconds : float or None
        Exposure time. Only used if input_in_electrons=False.
    mask : 2D array or None
        Mask of valid pixels. Convention here:
            True / 1  = use pixel
            False / 0 = ignore pixel
    corner : float or int
        If float < 1, fraction of image size used for each corner.
        If int, number of pixels in each corner.
    input_in_electrons : bool
        If True, data_cutout and err_map are already in e-.
        If False, they are assumed to be in e-/s and multiplied by exptime_seconds.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    data_sky_sub : 2D array
        Sky-subtracted data in e-.
    noise_map : 2D array
        Noise map in e-.
    diagnostics : dict
        Quality-control values.
    """

    import numpy as np

    data_cutout = np.atleast_2d(data_cutout).astype(float)
    err_map = np.atleast_2d(err_map).astype(float)

    if data_cutout.shape != err_map.shape:
        raise ValueError("data_cutout and err_map must have the same shape.")

    # --------------------------------------------------
    # 1. Keep / convert to electrons
    # --------------------------------------------------
    if input_in_electrons:
        data_counts = data_cutout.copy()
        err_counts = err_map.copy()
    else:
        if exptime_seconds is None:
            raise ValueError("exptime_seconds is required if input_in_electrons=False.")

        data_counts = data_cutout * exptime_seconds
        err_counts = err_map * exptime_seconds

    if mask is None:
        valid_mask = np.ones_like(data_counts, dtype=bool)
    else:
        valid_mask = np.asarray(mask).astype(bool)

    valid_mask &= np.isfinite(data_counts)
    valid_mask &= np.isfinite(err_counts)

    # --------------------------------------------------
    # 3. Corner sky pixels
    # --------------------------------------------------
    ny, nx = data_counts.shape

    if isinstance(corner, float) and corner < 1:
        cy = max(1, int(corner * ny))
        cx = max(1, int(corner * nx))
    else:
        cy = cx = int(corner)

    corner_mask = np.zeros_like(data_counts, dtype=bool)

    corner_mask[:cy, :cx] = True
    corner_mask[:cy, -cx:] = True
    corner_mask[-cy:, :cx] = True
    corner_mask[-cy:, -cx:] = True

    sky_mask = corner_mask & valid_mask

    corners = data_counts[sky_mask]
    print(corners.shape)
    if corners.size < 10:
        raise ValueError(
            f"Too few valid corner pixels for sky estimate: {corners.size}. "
            "Check mask or increase corner size."
        )

    # --------------------------------------------------
    # 4. Estimate constant sky in e-
    # --------------------------------------------------
    sky_level = np.nanmedian(corners)

    data_sky_sub = data_counts - sky_level

    # --------------------------------------------------
    # 5. Robust empirical sky noise from corners
    # --------------------------------------------------
    corner_resid = data_sky_sub[sky_mask]

    mad = np.nanmedian(np.abs(corner_resid - np.nanmedian(corner_resid)))
    sigma_sky = 1.4826 * mad
    sigma2_sky = sigma_sky**2

    # --------------------------------------------------
    # 6. Noise map
    # --------------------------------------------------
    has_err_map = np.any(np.isfinite(err_counts) & (err_counts > 0))

    if has_err_map:
        # If err_map is already a real noise map, combine with empirical sky scatter.
        # Be careful: this may double-count sky if err_map already includes it.
        noise_map = np.sqrt(err_counts**2 + sigma2_sky)
    else:
        # Fallback Poisson estimate.
        # Use sky-included counts for shot noise.
        signal_for_shot = data_sky_sub + sky_level
        sigma2_shot = np.clip(signal_for_shot, sky_level, None)
        noise_map = np.sqrt(sigma2_shot + sigma2_sky)

    # --------------------------------------------------
    # 7. Diagnostics on sky corners only
    # --------------------------------------------------
    mean_resid = np.nanmean(corner_resid)
    median_resid = np.nanmedian(corner_resid)
    std_resid = np.nanstd(corner_resid)

    median_noise_corner = np.nanmedian(noise_map[sky_mask])

    rms_ratio = std_resid / sigma_sky if sigma_sky > 0 else np.nan

    z = corner_resid / noise_map[sky_mask]
    z_mean = np.nanmean(z)
    z_median = np.nanmedian(z)
    z_std = np.nanstd(z)

    offset_ok = np.abs(median_resid) < 0.05 * std_resid
    rms_ok = 0.8 < rms_ratio < 1.25
    z_ok = 0.8 < z_std < 1.25

    is_fine = offset_ok and rms_ok and z_ok

    diagnostics = {
        "sky_level_e": sky_level,
        "sigma_sky_e": sigma_sky,
        "sigma2_sky_e2": sigma2_sky,
        "mean_resid_e": mean_resid,
        "median_resid_e": median_resid,
        "std_resid_e": std_resid,
        "median_noise_corner_e": median_noise_corner,
        "rms_ratio_std_over_sigma_sky": rms_ratio,
        "z_mean": z_mean,
        "z_median": z_median,
        "z_std": z_std,
        "n_corner_pixels": corners.size,
        "is_fine": is_fine,
    }

    if verbose:
        print("Single-cutout sky diagnostics")
        print("-----------------------------")
        print(f"sky_level              = {sky_level:.4g} e-")
        print(f"sigma_sky              = {sigma_sky:.4g} e-")
        print(f"mean corner residual   = {mean_resid:.4g} e-")
        print(f"median corner residual = {median_resid:.4g} e-")
        print(f"std corner residual    = {std_resid:.4g} e-")
        print(f"median corner noise    = {median_noise_corner:.4g} e-")
        print(f"std / sigma_sky        = {rms_ratio:.3f}")
        print(f"z mean / median / std  = {z_mean:.3f}, {z_median:.3f}, {z_std:.3f}")
        print(f"valid corner pixels    = {corners.size}")

        if is_fine:
            print("OK: sky subtraction looks fine.")
        else:
            print("WARNING: sky subtraction may not be fine.")

            if not offset_ok:
                print("- Corner residuals are not centered close enough to zero.")

            if not rms_ok:
                print("- Empirical corner RMS and robust sky RMS differ significantly.")

            if not z_ok:
                print("- Normalized corner residuals are not close to unit width.")

    return data_sky_sub, noise_map, diagnostics



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