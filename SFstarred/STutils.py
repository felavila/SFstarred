import numpy as np 
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u
from astropy.stats import sigma_clipped_stats

def jwst_ab_zeropoint_rate(header, photmjsr_key="PHOTMJSR"):
    """
    Return the JWST AB zeropoint for an image in e-/s.

    This assumes the original JWST image was in MJy/sr and was converted as:

        data_e_per_s = data_MJy_sr / PHOTMJSR

    Parameters
    ----------
    header : astropy.io.fits.Header
        JWST SCI or PRIMARY header containing PHOTMJSR and PIXAR_SR.

    photmjsr_key : str, optional
        Header keyword for the photometric conversion factor.

    Returns
    -------
    zp_ab : float
        AB zeropoint for fluxes in e-/s.
    """
    if photmjsr_key not in header:
        raise KeyError(f"Header needs {photmjsr_key}.")

    if "PIXAR_SR" not in header:
        raise KeyError("Header needs PIXAR_SR.")

    photmjsr = float(header[photmjsr_key])  # MJy/sr per e-/s
    pixar_sr = float(header["PIXAR_SR"])    # sr / pixel

    # e-/s -> MJy/sr -> Jy/pixel
    #
    # f_nu[Jy] = flux[e-/s] * PHOTMJSR * 1e6 * PIXAR_SR
    zp_ab = 8.90 - 2.5 * np.log10(photmjsr * 1e6 * pixar_sr)

    return zp_ab

def hst_ab_zeropoint(header):
    """
    Return the HST AB zeropoint for an image in e-/s.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Header containing either ABMAG or PHOTFLAM and PHOTPLAM.

    Returns
    -------
    zp_ab : float
        AB zeropoint for count-rate / e-/s images.
    """
    if "ABMAG" in header:
        return float(header["ABMAG"])

    if "PHOTFLAM" in header and "PHOTPLAM" in header:
        photflam = float(header["PHOTFLAM"])
        photplam = float(header["PHOTPLAM"])

        zp_ab = -2.5 * np.log10(photflam) - 5.0 * np.log10(photplam) - 2.408
        return zp_ab

    raise KeyError("Header needs either ABMAG or PHOTFLAM + PHOTPLAM.")



def read_jwst_for_starred(
    input_file,
    use_dq=True,
    dq_good_value=0,
    add_poisson_from_sci=True,
    exptime_key_candidates=("EFFEXPTM", "EXPTIME", "XPOSURE", "TEXPTIME"),
    photmjsr_key="PHOTMJSR",
):
    """
    Read a JWST drizzled image and build noise products for STARRED.

    All outputs are in units of e-/s (electrons per second).

    The conversion from the native MJy/sr uses the PHOTMJSR keyword
    (MJy/sr per e-/s), which must be present in the SCI or primary header.

    Parameters
    ----------
    input_file : str
        JWST drizzled FITS file (*_i2d.fits). Must contain SCI and ERR
        extensions in MJy/sr.

    output_sigma_file : str, optional
        If given, write the 1-sigma noise map (e-/s) to this path.

    use_dq : bool, optional
        If True, mask pixels with DQ != dq_good_value.

    dq_good_value : int, optional
        DQ flag value considered good (default 0).

    add_poisson_from_sci : bool, optional
        If True, add a Poisson variance term estimated from the science
        image in e-/s space:

            var_poisson [(e-/s)^2] = max(sci [e-/s], 0) / exptime [s]

        which is the standard shot-noise formula for a rate image.
        Only use this if you know ERR does not already include source
        Poisson noise (uncommon for _i2d products).

    exptime_key_candidates : tuple of str, optional
        Header keywords for effective exposure time in seconds.

    photmjsr_key : str, optional
        Header keyword for the photometric conversion factor
        (MJy/sr per e-/s). Required.

    Returns
    -------
    sci : 2-D ndarray, float, e-/s
        Science image. Bad/masked pixels are NaN.

    sigma : 2-D ndarray, float, e-/s
        1-sigma noise map in the same units as sci.
        Bad/masked pixels are NaN.

    sigma2 : 2-D ndarray, float, (e-/s)^2
        Variance map. Bad/masked pixels are NaN.

    exptime_seconds : float or None
        Effective exposure time found in the header, or None.
    """
    with fits.open(input_file) as hdul:
        sci_header     = hdul["SCI"].header.copy()
        primary_header = hdul[0].header.copy()

        data = hdul["SCI"].data.astype(float)   # MJy/sr
        err  = hdul["ERR"].data.astype(float)   # MJy/sr

        # --- photometric conversion factor -------------------------------
        photmjsr = None
        for hdr in (sci_header, primary_header):
            if photmjsr_key in hdr:
                photmjsr = float(hdr[photmjsr_key])
                break
        if photmjsr is None or photmjsr <= 0:
            raise ValueError(
                f"'{photmjsr_key}' not found or non-positive in FITS header. "
                "Cannot convert MJy/sr to e-/s."
            )

        # --- convert to e-/s --------------------------------------------
        data = data / photmjsr   # e-/s
        err  = err  / photmjsr   # e-/s

        # --- good-pixel mask --------------------------------------------
        good = (
            np.isfinite(data)
            & np.isfinite(err)
            & (err > 0)
        )
        if use_dq and "DQ" in hdul:
            good &= hdul["DQ"].data == dq_good_value

        # --- base variance in (e-/s)^2 ----------------------------------
        sigma2 = err ** 2

        # --- exposure time ----------------------------------------------
        exptime_seconds = None
        for key in exptime_key_candidates:
            for hdr in (sci_header, primary_header):
                if key in hdr:
                    exptime_seconds = float(hdr[key])
                    break
            if exptime_seconds is not None:
                break

        # --- optional Poisson correction --------------------------------
        # In e-/s space the shot-noise formula is exact and simple:
        #   var_poisson [(e-/s)^2] = rate [e-/s] / exptime [s]
        # because N_e = rate * t  ->  var(N_e) = N_e  ->  var(rate) = N_e / t^2
        #                                                             = rate / t
        # if add_poisson_from_sci:
        #     if exptime_seconds is None:
        #         raise ValueError(
        #             "add_poisson_from_sci=True requires exposure time in header."
        #         )
        #     positive_rate = np.clip(data, 0.0, None)   # e-/s
        #     poisson_var   = positive_rate / exptime_seconds  # (e-/s)^2
        #     sigma2        = sigma2 + poisson_var

        # --- assemble output arrays ------------------------------------
        sci    = data#np.where(good, data,            np.nan)
        sigma2 = sigma2#np.where(good, sigma2,          np.nan)
        #sigma  = np.where(good, np.sqrt(sigma2), np.nan)

        wcs = WCS(sci_header)
        zp_ab=jwst_ab_zeropoint_rate(sci_header)
        pixscale = proj_plane_pixel_scales(wcs) * u.deg
        pixscale_arcsec = pixscale.to(u.arcsec)

        #print(pixscale_arcsec)
        print("Mean pixel scale:", pixscale_arcsec.mean())
    return sci, sigma2, exptime_seconds, pixscale_arcsec.mean(),zp_ab,wcs

def read_hst_for_starred(
    input_file,
    output_sigma_file=None,
    add_poisson_from_sci=True,
    exptime_key_candidates=("EXPTIME", "TEXPTIME", "EFFEXPTM"),
):
    """
    Read an HST WFC3/IR HLA skycell drizzled mosaic (_drz.fits) and build
    noise products for STARRED.

    This reader targets the HLA skycell pipeline format:
        Extensions: PRIMARY, SCI, WHT, CTX, HDRTAB  (no ERR extension)

    Science units
    -------------
    SCI is in electrons/s (e-/s).  HLA skycell products are already
    gain-corrected before drizzling, so no additional gain conversion
    is needed.

    Noise model
    -----------
    The WHT extension from AstroDrizzle (IVM weighting) stores the
    inverse variance of the background per pixel:

        sigma2_background [(e-/s)^2] = 1 / WHT

    Per-pixel Poisson variance (optional) is then added:

        sigma2_poisson [(e-/s)^2] = max(sci [e-/s], 0) / exptime_pixel [s]

    where the per-pixel effective exposure time is recovered as:

        exptime_pixel [s] = WHT / median(WHT) * exptime_header [s]

    Parameters
    ----------
    input_file : str
        HST HLA skycell drizzled FITS (*_drz.fits).
        Required extensions: SCI, WHT.

    output_sigma_file : str, optional
        If given, write the 1-sigma noise map (e-/s) to this path.

    add_poisson_from_sci : bool, optional
        Add per-pixel Poisson variance estimated from the science image.
        Set False if you only want the background noise from WHT.

    exptime_key_candidates : tuple of str, optional
        Header keywords searched (in order) for total exposure time [s].

    Returns
    -------
    sci : 2-D ndarray, float, e-/s
        Science image. Bad/masked pixels are NaN.

    sigma : 2-D ndarray, float, e-/s
        1-sigma noise map. Bad/masked pixels are NaN.

    sigma2 : 2-D ndarray, float, (e-/s)^2
        Variance map. Bad/masked pixels are NaN.

    exptime_seconds : float or None
        Total effective exposure time found in the header, or None.
    """
    with fits.open(input_file) as hdul:
        sci_header     = hdul["SCI"].header.copy()
        primary_header = hdul[0].header.copy()

        data = hdul["SCI"].data.astype(float)   # e-/s
        wht  = hdul["WHT"].data.astype(float)   # inverse-variance weight map

        # --- exposure time from header ----------------------------------
        exptime_seconds = None
        for key in exptime_key_candidates:
            for hdr in (sci_header, primary_header):
                if key in hdr:
                    val = hdr[key]
                    if val is not None and float(val) > 0:
                        exptime_seconds = float(val)
                        break
            if exptime_seconds is not None:
                break

        # --- good-pixel mask --------------------------------------------
        # WHT == 0 means no coverage; also catch non-finite science values
        good = (
            np.isfinite(data)
            & (wht > 0)
        )

        # --- base variance from WHT -------------------------------------
        # AstroDrizzle IVM: WHT = 1 / sigma2_background
        sigma2 = np.where(good, 1.0 / np.where(wht > 0, wht, np.inf), np.nan)

        # --- per-pixel effective exposure time --------------------------
        # Needed for Poisson term.  The median over covered pixels gives
        # a robust normalisation to the total exptime from the header.
        if add_poisson_from_sci:
            if exptime_seconds is None:
                raise ValueError(
                    "add_poisson_from_sci=True requires an exposure time keyword "
                    f"in the header. Tried: {exptime_key_candidates}."
                )
            wht_covered   = wht[good]
            wht_median    = np.median(wht_covered) if wht_covered.size > 0 else 1.0
            exptime_pixel = (wht / wht_median) * exptime_seconds   # [s], per pixel

            positive_rate = np.clip(data, 0.0, None)               # e-/s
            safe_exptime  = np.where(exptime_pixel > 0, exptime_pixel, np.inf)
            poisson_var   = positive_rate / safe_exptime            # (e-/s)^2

            sigma2 = np.where(good, sigma2 + poisson_var, np.nan)

        # --- assemble output arrays ------------------------------------
        sci    = np.where(good, data,            np.nan)
        sigma  = np.where(good, np.sqrt(sigma2), np.nan)
        sigma2 = np.where(good, sigma2,          np.nan)

    wcs = WCS(sci_header)
    zp_ab=hst_ab_zeropoint(sci_header)
    pixscale = proj_plane_pixel_scales(wcs) * u.deg
    pixscale_arcsec = pixscale.to(u.arcsec)

    return sci, sigma2, exptime_seconds, pixscale_arcsec.mean(),zp_ab,wcs


def read_data_and_weight(path_data,path_weight):

    sci = fits.open(path_data)
    wht = fits.open(path_weight)
    h0 = sci[0].header
    TARGNAME = h0.get("TARGNAME")
    RA = h0.get("RA_TARG")
    DEC = h0.get("DEC_TARG")
    EXPTIME = h0.get("EXPTIME")
    FILTER  = h0.get("FILTER")
    print("Units of the image",h0["BUNIT"])
    wcs = WCS(h0)
    zp_ab=hst_ab_zeropoint(h0)
    pixscale = proj_plane_pixel_scales(wcs) * u.deg
    pixscale_arcsec = pixscale.to(u.arcsec).mean()
    data = sci[0].data
    positive_rate = np.clip(data, 0.0, None)   # e-/s
    poisson_var   = positive_rate / EXPTIME  # (e-/s)^2
    sigma2 = 1/wht[0].data + poisson_var
    mean, median, std = sigma_clipped_stats(data, sigma=5.0)
    return  RA,DEC,EXPTIME,FILTER,mean,median,std,wcs,pixscale_arcsec, zp_ab,data,sigma2,EXPTIME