import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from astropy.wcs.utils import proj_plane_pixel_scales
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.nddata import Cutout2D
from matplotlib.colors import SymLogNorm, LogNorm, Normalize, TwoSlopeNorm
from astropy.visualization import simple_norm
from matplotlib.colors import ListedColormap


def plot_stars_features(stars2D_data,starst2D_sigma2,radec=[],percentile=(1,94),index=[]):
	if len(index) != stars2D_data.shape[0]:
		index = range(stars2D_data.shape[0])
	for i in range(len(stars2D_data)):
		fig, axes = plt.subplots(1, 5, figsize=(22, 3),gridspec_kw={"wspace": 0.05},)
		if len(radec) == stars2D_data.shape[0]:
			print("ra, dec:", radec[i])
		ax = axes[0]
		ax.set_title(f"Star cutout {index[i]} ({i})")

		data = stars2D_data[i]
		norm = simple_norm(data, "sqrt", percent=99)
		#vmin,vmax = np.percentile(data,percentile)
		im = ax.imshow(data,norm=norm)
		fig.colorbar(im, ax=ax)

		# -------------------------
		# Weight / exposure map
		# -------------------------
		ax = axes[1]
		ax.set_title("Weight map = 1 / sigma2")

		sigma2 = starst2D_sigma2[i]

		weight = np.full_like(sigma2, np.nan, dtype=float)
		good = np.isfinite(sigma2) & (sigma2 > 0)
		weight[good] = 1.0 / sigma2[good]
		norm = simple_norm(weight, "sqrt", percent=99)
		im = ax.imshow(weight,norm=norm)
		fig.colorbar(im, ax=ax)

		# -------------------------
		# Sigma
		# -------------------------
		ax = axes[2]
		ax.set_title("sigma")
		
		sigma = np.full_like(sigma2, np.nan, dtype=float)
		sigma[good] = np.sqrt(sigma2[good])
		norm = simple_norm(sigma, "sqrt", percent=99)
			
		im = ax.imshow(sigma,norm=norm )
		fig.colorbar(im, ax=ax)

		# -------------------------
		# Sigma squared
		# -------------------------
		ax = axes[3]
		ax.set_title("sigma2")
		norm = simple_norm(sigma2, "sqrt", percent=99)
		im = ax.imshow(sigma2,norm=norm)
		fig.colorbar(im, ax=ax)

		ax = axes[4]
		ax.set_title("data/sigma")
		plot4 = data/sigma
		norm = simple_norm(plot4, "sqrt", percent=99)
		im = ax.imshow(plot4,norm=norm)
		fig.colorbar(im, ax=ax)
  
		# for ax in axes:
		# 	ax.set_xlabel("X [pix]")
		# 	ax.set_ylabel("Y [pix]")

		#fig.tight_layout()
		plt.show()

def twoplot(data,sigma2,what2plot=""):
	#stars2D_data = self.stars2D_data
	#starst2D_sigma2=self.starst2D_sigma2
	vmin,vmax = np.percentile(data,(1,95))
	for i in range(len(data)):
		fig, axes = plt.subplots(1, 2, figsize=(8, 3))
		ax = axes[0]
		#print("ra,dec",self.ra_dec[i])
		ax.set_title(f"Star cutout {i}")
		im = ax.imshow(data[i], cmap='gray_r', vmin=vmin,vmax=vmax)
		fig.colorbar(im, ax=ax)
		ax = axes[1]
		ax.set_title(what2plot)
		im = plt.imshow(sigma2[i])
		fig.colorbar(im, ax=ax)
		fig.tight_layout()
		# if save:
		#     plt.savefig(os.path.join(path_filter,f"star_cutout_{i+1}_{sky_coord.to_string(style='hmsdms', precision=2)}_{FILTER}.jpg"))
		plt.show()
            

def plot_lens_and_noisemap(
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

    norm = simple_norm(data, "sqrt", percent=99)
    im0 = ax[0].imshow(
        data,
        origin="lower",
        interpolation="nearest",
       # cmap=cmap_base,
        norm=norm
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

def add_en_arrows(
	ax,
	data,
	wcs,
	anchor=(0.75, 0.18),
	length_arcsec=1.0,
	color="white",
	fontsize=12,
	lw=2,):
	
	"""
	Add East and North arrows using the WCS.

	Parameters
	----------
	ax : matplotlib.axes.Axes
		Axis where the image is plotted.

	data : ndarray
		2D image data.

	wcs : astropy.wcs.WCS
		Celestial WCS of the image.

	anchor : tuple, optional
		Anchor position in fractional image coordinates (xfrac, yfrac).

	length_arcsec : float, optional
		Length of each compass arrow in arcsec.

	color : str, optional
		Arrow and label color.

	fontsize : int, optional
		Font size of the labels.

	lw : float, optional
		Arrow line width.
	"""
	ny, nx = data.shape

	# Anchor point in pixel coordinates
	x0 =  int(nx *0.9)
	y0 = int(0.2*ny)
	sky0 = wcs.pixel_to_world(x0, y0)

	# Move on the sky: North = PA 0 deg, East = PA 90 deg
	sky_north = sky0.directional_offset_by(0 * u.deg, length_arcsec * u.arcsec)
	sky_east = sky0.directional_offset_by(90 * u.deg, length_arcsec * u.arcsec)

	# Back to pixel coordinates
	xN, yN = wcs.world_to_pixel(sky_north)
	xE, yE = wcs.world_to_pixel(sky_east)
	cx = 0
	cy = 0
	if xE-x0<0:
		cx = data.shape[0]*0.02
	if yE-x0<0:
		cy = data.shape[0]*0.02
		#print(cx)
	#print(xE-x0, yE-y0)
	#print(xN-x0, yN-y0)
	ax.annotate("",xy=(xN, yN),xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), annotation_clip=False)
	ax.text(xN,yN, "N", color=color,fontsize=fontsize, ha="right", va="center", clip_on=False,)

	
	ax.annotate("",xy=(xE, yE),xytext=(x0, y0),arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),annotation_clip=False,)
	ax.text(xE,yE,"E",color=color,fontsize=fontsize,ha="center",va="center",clip_on=False,)


def plot_image_with_scalebar(
	data,
	wcs,
	title=None,
	scalebar_arcsec=1.0,
	percentiles=(1, 99),
	cmap="inferno",
	add_compass=True,
	compass_anchor=(0.75, 0.18),
	#compass_length_arcsec=0.5,
	compass_color="white",
	compass_fontsize=12,
	compass_lw=2,
	add_colorbar=False,
	colorbar_label="Flux",
	target_name = "",
	filter = "",
	figsize = (6, 6),
	norm=None
):
	"""
	Plot an astronomical image with an arcsec scale bar, optional E/N compass,
	and optional colorbar.

	Parameters
	----------
	data : ndarray
		2D image data.

	wcs : astropy.wcs.WCS
		WCS solution for the image.

	title : str, optional
		Plot title.

	scalebar_arcsec : float, optional
		Length of the scale bar in arcsec.

	percentiles : tuple, optional
		Percentile limits for display scaling.

	cmap : str, optional
		Colormap for the image.

	add_compass : bool, optional
		If True, draw the East/North compass.

	compass_anchor : tuple, optional
		Compass anchor in fractional image coordinates.

	compass_length_arcsec : float, optional
		Length of the compass arrows in arcsec. This is independent of the
		scale bar size.

	compass_color : str, optional
		Color of the compass arrows and labels.

	compass_fontsize : int, optional
		Font size of compass labels.

	compass_lw : float, optional
		Line width of compass arrows.

	add_colorbar : bool, optional
		If True, add a colorbar.

	colorbar_label : str, optional
		Label for the colorbar.
	"""
	wcs_cel = wcs.celestial

	vmin, vmax = np.nanpercentile(data, percentiles)

	fig, ax = plt.subplots(figsize=figsize)
	fig.canvas.draw()
	if norm:
		im = ax.imshow(data,origin="lower",cmap=cmap,norm=norm)
	else:
		im = ax.imshow(data,origin="lower",cmap=cmap,vmin=vmin,vmax=vmax)
	ax.set_xticks([])
	ax.set_yticks([])

	if title is not None:
		ax.set_title(title, fontsize=14)

	pixscale = proj_plane_pixel_scales(wcs_cel) * u.deg
	pixscale_arcsec = np.mean(pixscale.to(u.arcsec).value)

	scalebar_pix = scalebar_arcsec / pixscale_arcsec
	compass_length_arcsec = int(data.shape[0]*0.1) *  pixscale_arcsec
 	
	x0 = 0.12 * data.shape[1]
	y0 = 0.10 * data.shape[0]

	ax.plot(
		[x0, x0 + scalebar_pix],
		[y0, y0],
		color="white",
		lw=3,
	)

	ax.text(
		x0 + scalebar_pix / 2,
		y0 + 0.04 * data.shape[0],
		f'{scalebar_arcsec:g}"',
		color="white",
		ha="center",
		va="bottom",
		fontsize=12,
	)

	if add_compass:
		add_en_arrows(ax=ax,data=data, wcs=wcs_cel, anchor=compass_anchor, length_arcsec=compass_length_arcsec, color=compass_color, fontsize=compass_fontsize, lw=compass_lw,)
	text = ""
	text += target_name
	text += "\n" + filter
  		#ax.text(0.2,0.95,target_name, color="white", ha="center",va="top",fontsize=12,transform=ax.transAxes)
	if len(text):
		ax.text(0.2,0.95,text, color="white", ha="center",va="top",fontsize=20,transform=ax.transAxes)
		
	
	if add_colorbar:
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="4%", pad=0.05)
		cbar = fig.colorbar(im, cax=cax)
		cbar.set_label(colorbar_label)

	plt.tight_layout()
	return fig, ax

def plot_cutouts(data, rpix):

    """
    Given a 2D array of values such as the image of a star, create a 
    figure with four different scalings. The size of the resulting cutout 
    is set by 'rpix' and the red markers show the center value of 'rpix'.

    Parameters
    ----------
    data : np.ndarray
        The np.ndarray containing the array of science data, generally 
        after being cropped to a subarray using the make_cutouts() function.
    rpix : int
        An integer number of pixels for the cutout length and width.

    Returns
    -------
    figure : matplotlib.figure.Figure
        A figure showing each star on linear, lognorm, and log10 scales.
    """
    figure, mysubplot = plt.subplots(1, 4, figsize=(11, 11), sharex=True, sharey=True)
    mysubplot[0].imshow(data, vmin=0.0, vmax=np.amax(data), origin='lower', aspect='equal')
    mysubplot[1].imshow(data, vmin=0.0, vmax=np.amax(data)/100.0, origin='lower', aspect='equal')
    vmin = float(np.maximum(np.percentile(data, 0.0), 1.0e-6)) # Incase vmin < 0.
    vmax = np.percentile(data, 98.5)
    mysubplot[2].imshow(data, norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower', aspect='equal')
    mysubplot[3].imshow(np.log10(data), origin='lower', aspect='equal')
    mysubplot[0].set_title(r'100% Max')
    mysubplot[1].set_title(r'1% Max')
    mysubplot[2].set_title('LogNorm')
    mysubplot[3].set_title('Log10')
    for idx in range(4):
        mysubplot[idx].scatter(rpix, rpix, c='red', marker='+')
    plt.tight_layout()
    plt.show()

    return figure

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

    return fig, 



def nice_psf_plot(narrow_psfs,colorlabel="Flux",percent=99,stretch="log",cmap="inferno"):
    
    fig, ax = plt.subplots(figsize=(5, 5))

    image = narrow_psfs
    #{‘linear’, ‘sqrt’, :ref: ‘power’, log’, ‘asinh’, ‘sinh’},
    norm = simple_norm(image, stretch=stretch, percent=percent)

    im = ax.imshow(
        image,
        cmap=cmap,
        origin="lower",
        aspect="equal",
        norm=norm,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right",size="5%",   pad=0.05,   )

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(colorlabel)

    plt.show()
def plot_star_mask(
    data,
    noise_map,
    mask,
    n=None,
    figsize=(12, 5),
    cmap_data="inferno",
    cmap_base="gray",
    mask_color="red",
    alpha_mask=0.75,
    eps=1e-30,
    vmin=None,
    vmax=None,
    origin="lower",
):
    """
    Plot log10(data / noise_map) and data with mask overlay side by side.

    Parameters
    ----------
    data : ndarray
        Image data.

    noise_map : ndarray
        Noise map with the same shape as `data`.

    mask : ndarray
        Boolean mask with the same shape as `data`.
        True = good pixel, False = masked pixel.

    n : int or str, optional
        Star index/name used in the title.

    figsize : tuple, optional
        Figure size.

    cmap_data : str, optional
        Colormap for the signal-to-noise image.

    cmap_base : str, optional
        Colormap for the base data image in the mask panel.

    mask_color : str, optional
        Color used to show masked pixels.

    alpha_mask : float, optional
        Transparency of the mask overlay.
        Larger values make the mask stronger.

    eps : float, optional
        Small value to avoid division by zero or log10 problems.

    vmin, vmax : float or None, optional
        Color limits for the log10(data / noise_map) image.

    origin : {"lower", "upper"}, optional
        Image origin convention.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.

    axes : ndarray
        Axes array.
    """
    data = np.asarray(data, dtype=float)
    noise_map = np.asarray(noise_map, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    #safe_noise = np.maximum(noise_map, eps)
    log_sn_map = np.log10(data / noise_map)
    norm = simple_norm(log_sn_map, "sqrt", percent=99)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    im0 = axes[0].imshow(
        log_sn_map,
        #origin=origin,
        #cmap=cmap_data,
        interpolation="nearest",
         norm=norm,
    )

    title = "log10(data / noise_map)"
    if n is not None:
        title = f"Star {n}: " + title

    axes[0].set_title(title)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Base data image
    norm = simple_norm(data, "sqrt", percent=90)
    im_base = axes[1].imshow(
        data,
        #origin=origin,
        interpolation="nearest",
       # cmap=cmap_base,
        norm=norm
    )

    # Overlay only masked pixels.
    # mask == True  -> good pixel
    # mask == False -> masked pixel
    masked_overlay = np.where(~mask, 1.0, np.nan)

    mask_cmap = ListedColormap([mask_color])

    im1 = axes[1].imshow(
        masked_overlay,
        #origin=origin,
        interpolation="nearest",
        cmap=mask_cmap,
        alpha=alpha_mask,
    )

    axes[1].set_title(f"Data + mask overlay, alpha={alpha_mask}")
    fig.colorbar(im_base, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")

    plt.tight_layout()
    plt.show()

    return fig, axes