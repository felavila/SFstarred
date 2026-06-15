
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