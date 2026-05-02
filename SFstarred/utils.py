import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
import pickle 
from astropy.io import fits 

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