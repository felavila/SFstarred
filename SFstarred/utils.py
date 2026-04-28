import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
import pickle 

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