from copy import deepcopy
import numpy as np


def normalize_data_error(
    data_cutout,
    err_map,
    exptime_seconds=None,
    mask=None,
    corner=0.1,
    input_in_electrons=False,
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

    return data_sky_sub, noise_map


def normalize_data_error_photutils(
    data_cutout,
    err_map=None,
    exptime_seconds=None,
    mask=None,
    input_in_electrons=False,
    box_size=16,
    filter_size=3,
    exclude_percentile=50,
    nsigma=3.0,
    npixels=5,
    source_dilate=8,
    center=None,
    protect_radius=None,
    sky_annulus=None,
    bkg_method="sextractor",
    noise_mode="photutils_rms",
    min_sky_pixels=50,
    verbose=True,
    return_masks=False,
):
    import numpy as np
    from scipy.ndimage import binary_dilation
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, SExtractorBackground, MedianBackground
    from photutils.segmentation import detect_sources

    data_cutout = np.atleast_2d(data_cutout).astype(float)

    if err_map is not None:
        err_map = np.atleast_2d(err_map).astype(float)
        if err_map.shape != data_cutout.shape:
            raise ValueError("data_cutout and err_map must have the same shape.")

    ny, nx = data_cutout.shape

    # --------------------------------------------------
    # 1. Convert to electrons if needed
    # --------------------------------------------------
    if input_in_electrons:
        data_counts = data_cutout.copy()
        err_counts = None if err_map is None else err_map.copy()
    else:
        if exptime_seconds is None:
            raise ValueError("exptime_seconds is required if input_in_electrons=False.")

        data_counts = data_cutout * exptime_seconds
        err_counts = None if err_map is None else err_map * exptime_seconds

    # --------------------------------------------------
    # 2. Valid mask
    # --------------------------------------------------
    if mask is None:
        valid_mask = np.ones_like(data_counts, dtype=bool)
    else:
        valid_mask = np.asarray(mask).astype(bool)

    valid_mask &= np.isfinite(data_counts)

    if err_counts is not None:
        valid_mask &= np.isfinite(err_counts)

    if np.sum(valid_mask) < min_sky_pixels:
        raise ValueError("Too few valid pixels.")

    # photutils convention:
    # True = masked / ignored
    bad_mask = ~valid_mask

    # --------------------------------------------------
    # 3. Background estimator
    # --------------------------------------------------
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)

    if bkg_method == "sextractor":
        bkg_estimator = SExtractorBackground(sigma_clip=sigma_clip)
    elif bkg_method == "median":
        bkg_estimator = MedianBackground(sigma_clip=sigma_clip)
    else:
        raise ValueError("bkg_method must be 'sextractor' or 'median'.")

    box_size_eff = int(box_size)
    box_size_eff = max(4, min(box_size_eff, ny, nx))

    filter_size_eff = int(filter_size)
    if filter_size_eff % 2 == 0:
        filter_size_eff += 1

    # --------------------------------------------------
    # Helper: robust global background fallback
    # --------------------------------------------------
    def robust_global_background(data, mask_bad):
        vals = data[~mask_bad]
        vals = vals[np.isfinite(vals)]

        if vals.size < min_sky_pixels:
            raise ValueError("Too few pixels for robust global background.")

        for _ in range(10):
            med = np.nanmedian(vals)
            mad = np.nanmedian(np.abs(vals - med))
            sig = 1.4826 * mad

            if not np.isfinite(sig) or sig <= 0:
                sig = np.nanstd(vals)

            if not np.isfinite(sig) or sig <= 0:
                break

            keep = np.abs(vals - med) < 3.0 * sig

            if keep.sum() == vals.size:
                break

            vals = vals[keep]

        bkg_level = np.nanmedian(vals)
        bkg_rms = np.nanstd(vals)

        bkg_map = np.full_like(data, bkg_level, dtype=float)
        rms_map = np.full_like(data, bkg_rms, dtype=float)

        return bkg_map, rms_map

    def run_background2d(data, mask_bad):
        try:
            bkg = Background2D(
                data,
                box_size=box_size_eff,
                mask=mask_bad,
                filter_size=filter_size_eff,
                sigma_clip=sigma_clip,
                bkg_estimator=bkg_estimator,
                exclude_percentile=exclude_percentile,
            )

            return bkg.background, bkg.background_rms, "Background2D"

        except ValueError as e:
            if verbose:
                print("WARNING: Background2D failed.")
                print(str(e))
                print("Using robust global background fallback.")

            bkg_map, rms_map = robust_global_background(data, mask_bad)
            return bkg_map, rms_map, "global_fallback"

    # --------------------------------------------------
    # 4. First-pass background
    # --------------------------------------------------
    bkg0_map, bkg0_rms_map, bkg0_mode = run_background2d(data_counts, bad_mask)

    data_sub0 = data_counts - bkg0_map

    # --------------------------------------------------
    # 5. Source detection
    # --------------------------------------------------
    threshold = nsigma * bkg0_rms_map

    try:
        segm = detect_sources(
            data_sub0,
            threshold,
            n_pixels=npixels,
            mask=bad_mask,
        )
    except TypeError:
        segm = detect_sources(
            data_sub0,
            threshold,
            npixels=npixels,
            mask=bad_mask,
        )

    if segm is None:
        source_mask = np.zeros_like(data_counts, dtype=bool)
    else:
        source_mask = segm.data > 0

    # --------------------------------------------------
    # 6. Protect main source
    # --------------------------------------------------
    if center is None:
        tmp = data_sub0.copy()
        tmp[bad_mask] = -np.inf
        cy, cx = np.unravel_index(np.nanargmax(tmp), tmp.shape)
    else:
        cy, cx = center

    yy, xx = np.indices(data_counts.shape)
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    if protect_radius is None:
        protect_radius = max(3, source_dilate)

    source_mask |= r <= protect_radius

    if source_dilate is not None and source_dilate > 0:
        source_mask = binary_dilation(source_mask, iterations=int(source_dilate))

    source_mask &= valid_mask

    # --------------------------------------------------
    # 7. Sky mask
    # --------------------------------------------------
    sky_mask = valid_mask & (~source_mask)

    if sky_annulus is not None:
        r_inner, r_outer = sky_annulus
        annulus_mask = (r >= r_inner) & (r <= r_outer)
        sky_mask &= annulus_mask
    else:
        annulus_mask = np.ones_like(valid_mask, dtype=bool)

    if np.sum(sky_mask) < min_sky_pixels:
        raise ValueError(
            f"Too few sky pixels after source masking: {np.sum(sky_mask)}. "
            "Try smaller source_dilate, larger cutout, or different sky_annulus."
        )

    final_mask = bad_mask | source_mask | (~annulus_mask)

    # --------------------------------------------------
    # 8. Final background
    # --------------------------------------------------
    sky_level_map, sky_rms_map, bkg_mode = run_background2d(data_counts, final_mask)

    data_sky_sub = data_counts - sky_level_map

    # --------------------------------------------------
    # 9. Noise map
    # --------------------------------------------------
    sky_resid = data_sky_sub[sky_mask]
    empirical_sky_std = np.nanstd(sky_resid)
    empirical_sky_median = np.nanmedian(sky_resid)

    if noise_mode == "photutils_rms":
        noise_map = sky_rms_map.copy()

    elif noise_mode == "errmap":
        if err_counts is None:
            raise ValueError("err_map is required for noise_mode='errmap'.")
        noise_map = err_counts.copy()

    elif noise_mode == "rescale_errmap":
        if err_counts is None:
            raise ValueError("err_map is required for noise_mode='rescale_errmap'.")

        med_err_sky = np.nanmedian(err_counts[sky_mask])

        if not np.isfinite(med_err_sky) or med_err_sky <= 0:
            raise ValueError("Invalid median err_map value in sky pixels.")

        scale = empirical_sky_std / med_err_sky
        noise_map = err_counts * scale

    elif noise_mode == "quadrature":
        if err_counts is None:
            raise ValueError("err_map is required for noise_mode='quadrature'.")

        noise_map = np.sqrt(err_counts**2 + sky_rms_map**2)

    else:
        raise ValueError(
            "noise_mode must be 'photutils_rms', 'errmap', "
            "'rescale_errmap', or 'quadrature'."
        )

    bad_noise = (~np.isfinite(noise_map)) | (noise_map <= 0)

    if np.any(bad_noise):
        fallback = np.nanmedian(noise_map[sky_mask])
        if not np.isfinite(fallback) or fallback <= 0:
            fallback = empirical_sky_std
        noise_map[bad_noise] = fallback

    # --------------------------------------------------
    # 10. Diagnostics
    # --------------------------------------------------
    z = data_sky_sub[sky_mask] / noise_map[sky_mask]

    z_mean = np.nanmean(z)
    z_median = np.nanmedian(z)
    z_std = np.nanstd(z)

    median_noise_sky = np.nanmedian(noise_map[sky_mask])
    median_bkg_rms = np.nanmedian(sky_rms_map[sky_mask])

    diagnostics = {
        "center_yx": (float(cy), float(cx)),
        "median_sky_level": float(np.nanmedian(sky_level_map[sky_mask])),
        "median_background_rms": float(median_bkg_rms),
        "empirical_sky_median": float(empirical_sky_median),
        "empirical_sky_std": float(empirical_sky_std),
        "median_noise_sky": float(median_noise_sky),
        "z_mean": float(z_mean),
        "z_median": float(z_median),
        "z_std": float(z_std),
        "n_valid_pixels": int(np.sum(valid_mask)),
        "n_source_pixels": int(np.sum(source_mask)),
        "n_sky_pixels": int(np.sum(sky_mask)),
        "box_size_eff": int(box_size_eff),
        "filter_size_eff": int(filter_size_eff),
        "exclude_percentile": float(exclude_percentile),
        "bkg_method": bkg_method,
        "bkg0_mode": bkg0_mode,
        "bkg_mode": bkg_mode,
        "noise_mode": noise_mode,
        "is_fine": bool(0.8 < z_std < 1.25),
    }

    if noise_mode == "rescale_errmap":
        diagnostics["errmap_rescale_factor"] = float(scale)

    if verbose:
        print("Photutils/SExtractor-like sky diagnostics")
        print("-----------------------------------------")
        print(f"detected center        = ({cy:.1f}, {cx:.1f}) pix")
        print(f"bkg method             = {bkg_method}")
        print(f"first bkg mode         = {bkg0_mode}")
        print(f"final bkg mode         = {bkg_mode}")
        print(f"noise mode             = {noise_mode}")
        print(f"box size               = {box_size_eff}")
        print(f"filter size            = {filter_size_eff}")
        print(f"exclude percentile     = {exclude_percentile}")
        print(f"median sky level       = {np.nanmedian(sky_level_map[sky_mask]):.4g}")
        print(f"median bkg RMS         = {median_bkg_rms:.4g}")
        print(f"empirical sky median   = {empirical_sky_median:.4g}")
        print(f"empirical sky std      = {empirical_sky_std:.4g}")
        print(f"median noise sky       = {median_noise_sky:.4g}")
        print(f"z mean / median / std  = {z_mean:.3f}, {z_median:.3f}, {z_std:.3f}")
        print(f"valid pixels           = {np.sum(valid_mask)}")
        print(f"source-mask pixels     = {np.sum(source_mask)}")
        print(f"sky pixels             = {np.sum(sky_mask)}")

        if noise_mode == "rescale_errmap":
            print(f"err_map scale factor   = {scale:.4g}")

        if 0.8 < z_std < 1.25:
            print("OK: noise normalization looks reasonable.")
        else:
            print("WARNING: normalized residuals are not close to unit width.")

    if return_masks:
        return data_sky_sub, noise_map, diagnostics, source_mask, sky_mask, sky_level_map

    return data_sky_sub, noise_map

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


import numpy as np


def subtract_background_centered_star(
    data_cutout,
    mask=None,
    source_radius=0.30,
    spike_angles=None,
    spike_width=2.0,
    sigma=3.5,
    maxiters=5,
    input_in_electrons=True,
    exptime_seconds=None,
    verbose=True,
):
    """
    Estimate and subtract a constant background from a centered-star cutout.

    Parameters
    ----------
    data_cutout : 2D array
        Image containing a star approximately at the center.

    mask : 2D bool array or None
        Valid-pixel mask:
            True  = use pixel
            False = ignore pixel

    source_radius : float
        Radius excluded around the star.

        If source_radius < 1, it is interpreted as a fraction of the
        smallest image dimension. For example, 0.30 excludes a radius
        equal to 30% of the cutout size.

        If source_radius >= 1, it is interpreted directly in pixels.

    spike_angles : sequence or None
        Angles of diffraction-spike lines in degrees, measured
        counterclockwise from the positive x direction.

        Examples:
            None             : no explicit spike masking
            (0, 90)          : horizontal and vertical spikes
            (45, 135)        : diagonal spikes
            (0, 45, 90, 135) : horizontal, vertical, and diagonal spikes

    spike_width : float
        Half-width of each masked spike strip in pixels.

    sigma : float
        Sigma threshold used for iterative clipping.

    maxiters : int
        Maximum number of clipping iterations.

    input_in_electrons : bool
        True if input data are already in electrons.
        False if input data are in electrons/second.

    exptime_seconds : float or None
        Exposure time required when input_in_electrons=False.

    verbose : bool
        Print background diagnostics.

    Returns
    -------
    data_sky_sub : 2D array
        Background-subtracted image in electrons.

    sigma_background : float
        Robust background RMS in electrons.

    background_mask : 2D bool array
        Pixels actually used to estimate the background.
    """

    data = np.asarray(data_cutout, dtype=float)

    if data.ndim != 2:
        raise ValueError("data_cutout must be a 2D array.")

    # ------------------------------------------------------------
    # Convert to electrons
    # ------------------------------------------------------------
    if input_in_electrons:
        data_e = data.copy()
    else:
        if exptime_seconds is None:
            raise ValueError(
                "exptime_seconds is required when input_in_electrons=False."
            )
        data_e = data * exptime_seconds

    ny, nx = data_e.shape

    # Geometric center of the cutout
    x_center = (nx - 1) / 2
    y_center = (ny - 1) / 2

    y, x = np.indices(data_e.shape)
    dx = x - x_center
    dy = y - y_center
    radius = np.hypot(dx, dy)

    # ------------------------------------------------------------
    # Initial valid-pixel mask
    # ------------------------------------------------------------
    valid = np.isfinite(data_e)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != data_e.shape:
            raise ValueError("mask and data_cutout must have the same shape.")

        valid &= mask

    # ------------------------------------------------------------
    # Exclude the central star and its extended core
    # ------------------------------------------------------------
    if source_radius < 1:
        radius_pixels = source_radius * min(ny, nx)
    else:
        radius_pixels = float(source_radius)

    background_mask = valid & (radius >= radius_pixels)

    # ------------------------------------------------------------
    # Exclude diffraction-spike strips
    # ------------------------------------------------------------
    if spike_angles is not None:
        for angle in spike_angles:
            theta = np.deg2rad(angle)

            # Perpendicular distance from each pixel to a line
            # passing through the center at angle theta.
            distance_to_spike = np.abs(
                -np.sin(theta) * dx + np.cos(theta) * dy
            )

            background_mask &= distance_to_spike > spike_width

    indices = np.flatnonzero(background_mask)
    values = data_e.ravel()[indices]

    if values.size < 20:
        raise ValueError(
            f"Only {values.size} background pixels remain. "
            "Reduce source_radius or spike_width."
        )

    # ------------------------------------------------------------
    # Iterative robust sigma clipping
    # ------------------------------------------------------------
    keep = np.ones(values.size, dtype=bool)

    for _ in range(maxiters):
        selected = values[keep]

        median = np.median(selected)
        mad = np.median(np.abs(selected - median))
        robust_std = 1.4826 * mad

        # Fallback when the MAD is zero
        if not np.isfinite(robust_std) or robust_std <= 0:
            robust_std = np.std(selected)

        if not np.isfinite(robust_std) or robust_std <= 0:
            break

        new_keep = np.abs(values - median) < sigma * robust_std

        if np.array_equal(new_keep, keep):
            break

        keep = new_keep

    background_values = values[keep]

    if background_values.size < 10:
        raise ValueError(
            "Too few pixels remain after sigma clipping. "
            "Reduce source_radius, spike_width, or sigma clipping strength."
        )

    # ------------------------------------------------------------
    # Final background and noise estimates
    # ------------------------------------------------------------
    sky_level = np.median(background_values)

    mad = np.median(np.abs(background_values - sky_level))
    sigma_background = 1.4826 * mad

    if sigma_background <= 0:
        sigma_background = np.std(background_values)

    data_sky_sub = data_e - sky_level

    # Final mask containing only pixels retained after clipping
    final_background_mask = np.zeros_like(background_mask)
    final_background_mask.ravel()[indices[keep]] = True

    if verbose:
        residuals = data_sky_sub[final_background_mask]

        print("Background diagnostics")
        print("----------------------")
        print(f"Central exclusion radius : {radius_pixels:.2f} pixels")
        print(f"Background level         : {sky_level:.5g} e-")
        print(f"Background RMS           : {sigma_background:.5g} e-")
        print(f"Median residual          : {np.median(residuals):.5g} e-")
        print(f"Residual standard dev.   : {np.std(residuals):.5g} e-")
        print(f"Background pixels used   : {background_values.size}")

    return data_sky_sub, sigma_background, final_background_mask


import numpy as np

from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.segmentation import (
    detect_sources,
    deblend_sources,
)
from scipy.ndimage import binary_dilation


def mask_sources_except_center(
    data,
    input_mask=None,
    center=None,
    central_search_radius=5,
    threshold_sigma=3.0,
    npixels=5,
    deblend=True,
    nlevels=32,
    contrast=0.001,
    filter_fwhm=2.0,
    dilation_radius=3,
    verbose=True,
):
    """
    Detect and mask every source except the main source near the image center.

    Parameters
    ----------
    data : 2D array
        Image data.

    input_mask : 2D bool array or None
        Pixels that must already be ignored.

        Convention:
            True  = masked / ignored
            False = usable

    center : tuple or None
        Main-source position as ``(x, y)`` in pixels.
        If None, the geometric center of the image is used.

    central_search_radius : float
        If no segmentation label exists exactly at the center, search for
        the nearest detected source within this radius.

    threshold_sigma : float
        Detection threshold above the estimated background.

    npixels : int
        Minimum number of connected pixels required for a source.

    deblend : bool
        Separate overlapping sources when possible.

    nlevels : int
        Number of deblending levels.

    contrast : float
        Minimum contrast used during deblending.

    filter_fwhm : float
        FWHM of the Gaussian kernel used before source detection.
        Set to 0 or None to disable filtering.

    dilation_radius : int
        Number of binary-dilation iterations applied to contaminating
        source masks. This masks their faint wings.

    verbose : bool
        Print detection information.

    Returns
    -------
    contaminant_mask : 2D bool array
        Mask of every detected source except the central main source.

        True  = contaminating source
        False = background or main source

    segmentation : SegmentationImage or None
        Final segmentation map.

    main_label : int or None
        Segmentation label assigned to the main source.
    """

    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError("data must be a 2D array.")

    ny, nx = data.shape

    if center is None:
        x_center = (nx - 1) / 2
        y_center = (ny - 1) / 2
    else:
        x_center, y_center = center

    if input_mask is None:
        invalid_mask = ~np.isfinite(data)
    else:
        input_mask = np.asarray(input_mask, dtype=bool)

        if input_mask.shape != data.shape:
            raise ValueError("input_mask and data must have the same shape.")

        invalid_mask = input_mask | ~np.isfinite(data)

    # ------------------------------------------------------------
    # Estimate the background robustly
    # ------------------------------------------------------------
    mean_sky, median_sky, sigma_sky = sigma_clipped_stats(
        data,
        mask=invalid_mask,
        sigma=3.0,
        maxiters=5,
    )

    if not np.isfinite(sigma_sky) or sigma_sky <= 0:
        raise ValueError("Could not estimate a valid background RMS.")

    threshold = median_sky + threshold_sigma * sigma_sky

    # ------------------------------------------------------------
    # Smooth the image for source detection
    # ------------------------------------------------------------
    detection_image = data - median_sky

    if filter_fwhm is not None and filter_fwhm > 0:
        kernel_sigma = filter_fwhm / 2.3548
        kernel_size = max(3, int(np.ceil(6 * kernel_sigma)))

        # Kernel dimensions should be odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = Gaussian2DKernel(
            x_stddev=kernel_sigma,
            x_size=kernel_size,
            y_size=kernel_size,
        )

        detection_image = convolve(
            detection_image,
            kernel,
            mask=invalid_mask,
            normalize_kernel=True,
        )

    # Because the background was subtracted, use sigma-only threshold
    detection_threshold = threshold_sigma * sigma_sky

    segmentation = detect_sources(
        detection_image,
        threshold=detection_threshold,
        npixels=npixels,
        mask=invalid_mask,
    )

    if segmentation is None:
        if verbose:
            print("No sources were detected.")

        return np.zeros_like(data, dtype=bool), None, None

    # ------------------------------------------------------------
    # Deblend overlapping sources
    # ------------------------------------------------------------
    if deblend and segmentation.nlabels > 1:
        try:
            segmentation = deblend_sources(
                detection_image,
                segmentation,
                npixels=npixels,
                nlevels=nlevels,
                contrast=contrast,
                progress_bar=False,
            )
        except ValueError:
            # Keep the original segmentation if deblending fails
            pass

    segment_data = segmentation.data

    # ------------------------------------------------------------
    # Find the main source
    # ------------------------------------------------------------
    x_index = int(round(x_center))
    y_index = int(round(y_center))

    x_index = np.clip(x_index, 0, nx - 1)
    y_index = np.clip(y_index, 0, ny - 1)

    main_label = int(segment_data[y_index, x_index])

    # If the exact center is background, find the nearest detected source
    if main_label == 0:
        y_grid, x_grid = np.indices(data.shape)

        search_region = (
            (x_grid - x_center) ** 2
            + (y_grid - y_center) ** 2
            <= central_search_radius**2
        )

        nearby_labels = np.unique(segment_data[search_region])
        nearby_labels = nearby_labels[nearby_labels != 0]

        if nearby_labels.size > 0:
            best_distance = np.inf
            best_label = None

            for label in nearby_labels:
                yy, xx = np.where(segment_data == label)

                source_x = np.mean(xx)
                source_y = np.mean(yy)

                distance = np.hypot(
                    source_x - x_center,
                    source_y - y_center,
                )

                if distance < best_distance:
                    best_distance = distance
                    best_label = int(label)

            main_label = best_label

    # ------------------------------------------------------------
    # Mask every segmentation label except the main source
    # ------------------------------------------------------------
    if main_label is None or main_label == 0:
        # No source associated with the expected central position.
        # Mask all detected sources.
        contaminant_mask = segment_data > 0

        if verbose:
            print(
                "WARNING: no central source was identified; "
                "all detected sources were masked."
            )
    else:
        contaminant_mask = (
            (segment_data > 0)
            & (segment_data != main_label)
        )

    # Expand masks to include faint source wings
    if dilation_radius > 0:
        contaminant_mask = binary_dilation(
            contaminant_mask,
            iterations=dilation_radius,
        )

    # Make absolutely sure the main segmentation region is restored
    if main_label is not None and main_label > 0:
        main_source_mask = segment_data == main_label
        contaminant_mask[main_source_mask] = False

    if verbose:
        n_sources = segmentation.nlabels
        n_contaminants = n_sources - int(
            main_label is not None and main_label > 0
        )

        print("Source-mask diagnostics")
        print("-----------------------")
        print(f"Background level      : {median_sky:.5g}")
        print(f"Background RMS        : {sigma_sky:.5g}")
        print(f"Detected sources      : {n_sources}")
        print(f"Main source label     : {main_label}")
        print(f"Contaminating sources : {max(0, n_contaminants)}")
        print(f"Masked pixels         : {contaminant_mask.sum()}")

    return ~contaminant_mask, segmentation, main_label