import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
import pickle 
from astropy.io import fits 
from multiprocessing import Pool, cpu_count
import os 
import pandas as pd 


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
  

def read_jwst_for_starred(
    input_file,
    output_sigma_file=None,
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
        if add_poisson_from_sci:
            if exptime_seconds is None:
                raise ValueError(
                    "add_poisson_from_sci=True requires exposure time in header."
                )
            positive_rate = np.clip(data, 0.0, None)   # e-/s
            poisson_var   = positive_rate / exptime_seconds  # (e-/s)^2
            sigma2        = sigma2 + poisson_var

        # --- assemble output arrays ------------------------------------
        sci    = np.where(good, data,            np.nan)
        sigma2 = np.where(good, sigma2,          np.nan)
        #sigma  = np.where(good, np.sqrt(sigma2), np.nan)

    # --- optional file output ------------------------------------------
    # if output_sigma_file is not None:
    #     hdu = fits.PrimaryHDU(sigma, header=sci_header)
    #     hdu.header["BUNIT"]    = "e-/s"
    #     hdu.header["PHOTMJSR"] = (photmjsr, "MJy/sr per e-/s used for conversion")
    #     hdu.header["COMMENT"]  = "1-sigma noise map in e-/s"
    #     if exptime_seconds is not None:
    #         hdu.header["EXPTUSED"] = (exptime_seconds, "Effective exposure time [s]")
    #     hdu.writeto(output_sigma_file, overwrite=True)

    return sci, sigma2, exptime_seconds

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

    # --- optional file output ------------------------------------------
    if output_sigma_file is not None:
        hdu = fits.PrimaryHDU(sigma, header=sci_header)
        hdu.header["BUNIT"]   = "e-/s"
        hdu.header["COMMENT"] = "1-sigma noise map derived from WHT extension"
        if exptime_seconds is not None:
            hdu.header["EXPTUSED"] = (exptime_seconds, "Total exposure time [s]")
        hdu.writeto(output_sigma_file, overwrite=True)

    return sci, sigma2, exptime_seconds

def plot_image_and_noisemap(
    data,
    noisemaps,
    image_positions,
    image_names=None,
    figsize=(12, 5),
    cmap_data="gray",
    cmap_noise="viridis",
    marker="x",
    marker_color="red",
    marker_size=80,
    text_color="white",
    vmin=None,
    vmax=None,
    noise_vmin=None,
    noise_vmax=None,
):
    """
    Plot the science image and the noise map side by side.

    Parameters
    ----------
    data : array-like
        2D science image.
    noisemaps : array-like
        2D noise map or variance map.
    image_positions : array-like
        Pixel positions of the lensed images. Expected shape is (N, 2),
        where columns are x and y pixel coordinates.
    image_names : list of str, optional
        Names/labels of the images, e.g. ["A", "B", "C", "D"].
    figsize : tuple, optional
        Figure size.
    cmap_data : str, optional
        Colormap for the science image.
    cmap_noise : str, optional
        Colormap for the noise map.
    marker : str, optional
        Marker style for image positions.
    marker_color : str, optional
        Marker color.
    marker_size : float, optional
        Marker size.
    text_color : str, optional
        Color of the image labels.
    vmin, vmax : float, optional
        Intensity limits for the science image.
    noise_vmin, noise_vmax : float, optional
        Intensity limits for the noise map.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : numpy.ndarray
        Array of axes.
    """

    data = np.asarray(data)
    noisemaps = np.asarray(noisemaps)
    image_positions = np.asarray(image_positions)

    if data.ndim != 2:
        raise ValueError(f"`data` must be 2D, but got shape {data.shape}")

    if noisemaps.ndim != 2:
        raise ValueError(f"`noisemaps` must be 2D, but got shape {noisemaps.shape}")

    if data.shape != noisemaps.shape:
        raise ValueError(
            f"`data` and `noisemaps` must have the same shape, "
            f"but got {data.shape} and {noisemaps.shape}"
        )

    if image_positions.ndim != 2 or image_positions.shape[1] != 2:
        raise ValueError(
            "`image_positions` must have shape (N, 2), with columns [x, y]."
        )

    if image_names is None:
        image_names = [str(i) for i in range(len(image_positions))]

    if len(image_names) != len(image_positions):
        raise ValueError(
            "`image_names` must have the same length as `image_positions`."
        )

    fig, ax = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    im0 = ax[0].imshow(
        data,
        origin="lower",
        cmap=cmap_data,
        vmin=vmin,
        vmax=vmax,
    )

    ax[0].scatter(
        image_positions[:, 0],
        image_positions[:, 1],
        marker=marker,
        s=marker_size,
        color=marker_color,
    )

    for name, (x, y) in zip(image_names, image_positions):
        ax[0].text(
            x + 2,
            y + 2,
            name,
            color=text_color,
            fontsize=12,
            weight="bold",
        )

    ax[0].set_title("Image")
    ax[0].set_xlabel("x [pix]")
    ax[0].set_ylabel("y [pix]")

    cbar0 = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Flux")

    im1 = ax[1].imshow(
        noisemaps,
        origin="lower",
        cmap=cmap_noise,
        vmin=noise_vmin,
        vmax=noise_vmax,
    )

    ax[1].set_title("Noise map")
    ax[1].set_xlabel("x [pix]")
    ax[1].set_ylabel("y [pix]")

    cbar1 = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Noise")

    return fig, ax



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
