from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.cm as cm
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm, Normalize, TwoSlopeNorm
import pandas as pd 
import pickle
from photutils.detection import DAOStarFinder, find_peaks
from photutils.aperture import CircularAperture
from photutils.centroids import centroid_sources, centroid_2dg, centroid_com
from photutils.aperture import (ApertureStats, CircularAnnulus,
                               CircularAperture, aperture_photometry)

from astropy.visualization import simple_norm
from matplotlib.patches import Polygon
#from matplotlib.path import Path
from skimage.measure import find_contours

import pyregion

from scipy.ndimage import binary_dilation


from SFstarred.STutils import read_jwst_for_starred,read_hst_for_starred,read_data_and_weight
from SFstarred.utils import find_nearest_object,create_rectangle_patch,make_cutouts
from SFstarred.plots import plot_image_with_scalebar,plot_stars_features,nice_psf_plot
from SFstarred.stars_cut import normalize_data_error,make_sigma2_from_hst_weight


tabla1 = pd.read_csv(Path(__file__).resolve().parent / "suportdata" / "tablea1.csv")
#TODO we have to check the headers and data ...
class DataStarred:
    
    def __init__(self, path_data,path_weight=None):
        path_file = Path(path_data)
        self.path_file = path_file
        if not path_file.is_file():
                raise FileNotFoundError(f"File not found: {path_file}")
        #path_weight = None
        if path_weight is not None:
            path_weight = Path(path_weight)
            if not path_weight.is_file():
                raise FileNotFoundError(f"File weight not found: {path_weight}")
            self.path_weight = path_weight
        else:
            self.path_weight = None
        
        self._readfits()
        self._detect_lens()

        #filter object 

    def _readfits(self):
        self.header0 = fits.open(self.path_file)[0].header
        data, self.header = fits.getdata(self.path_file, header=True)
        self.TELESCOP = self.header0.get("TELESCOP")
        if  self.TELESCOP=="JWST":
            self.TARGNAME = self.header0.get("TARGPROP")
            self.RA = self.header0.get("TARG_RA")
            self.DEC = self.header0.get("TARG_DEC")
            self.FILTER  = self.header0.get("FILTER")
            self.data, self.sigma2, self.exptime_seconds,pixel_scale,zp_ab,wcs = read_jwst_for_starred(self.path_file,add_poisson_from_sci=True)
            self.mean, self.median, self.std = sigma_clipped_stats(self.data, sigma=5.0)
            pixel_scale = pixel_scale.value
        
        else:
            if self.path_weight is not None:
                self.RA,self.DEC, self.EXPTIME,self.FILTER,self.mean, self.median, self.std,wcs,pixel_scale, zp_ab,self.data,self.sigma2,self.exptime_seconds = read_data_and_weight(self.path_file,self.path_weight)
            else:
                self.TARGNAME = self.header.get("TARGNAME")
                self.RA = self.header0.get("RA_TARG")
                self.DEC = self.header0.get("DEC_TARG")
                self.EXPTIME = self.header0.get("EXPTIME")
                self.FILTER = self.header0.get("FILTER")
                self.data, self.sigma2, self.exptime_seconds,pixel_scale,zp_ab,wcs = read_hst_for_starred(self.path_file,add_poisson_from_sci=True)
                self.mean, self.median, self.std = sigma_clipped_stats(self.data, sigma=5.0)
                pixel_scale = pixel_scale.value
            
        self.pixel_scale = pixel_scale
        self.zp_ab = zp_ab
        self.wcs = wcs
        self.default_fwhm = 5
        self.default_threshold = 50.0 * self.std
        if self.FILTER == "F160W":
            self.default_fwhm = 5
            self.default_threshold = 50.0 * self.std

        elif self.FILTER == "F814W":
            self.default_fwhm = 5
            self.default_threshold = 10.0 * self.std

        elif self.FILTER == "F475X":
            self.default_fwhm = 5
            self.default_threshold = 10.0 * self.std
    def from_raw_to_clean(
        self,
        box_size=64,
        verbose=True,
        plot=True,
        dilate_iter=3,
        add_bkg_rms=False,
        inplace=False,
    ):
        """
        Background-subtract image and return cleaned data + noise map in electrons.

        Assumes input:
            self.data   in e-/s
            self.sigma2 in (e-/s)^2

        Output:
            data_clean_e  in e-
            noise_clean_e in e-
            sigma2_clean_e in e-^2
        """

        import numpy as np
        from astropy.stats import SigmaClip, sigma_clipped_stats
        from photutils.background import Background2D, MedianBackground
        from scipy.ndimage import binary_dilation

        ets = self.exptime_seconds

        # --------------------------------------------------
        # Convert raw image and variance to electrons
        # --------------------------------------------------
        data_e = self.data * ets
        sigma2_e = self.sigma2 * ets**2

        # Avoid invalid variance values
        sigma2_e = np.where(np.isfinite(sigma2_e) & (sigma2_e > 0), sigma2_e, np.nan)

        # --------------------------------------------------
        # Source mask
        # --------------------------------------------------
        med, _, std = sigma_clipped_stats(data_e, sigma=3.0)

        source_mask = data_e > (med + 3.0 * std)
        source_mask = binary_dilation(source_mask, iterations=dilate_iter)

        empty = (~source_mask) & np.isfinite(data_e)

        # --------------------------------------------------
        # Background model
        # --------------------------------------------------
        bkg = Background2D(
            data_e,
            box_size=box_size,
            filter_size=3,
            sigma_clip=SigmaClip(sigma=3.0),
            bkg_estimator=MedianBackground(),
            mask=source_mask,
            exclude_percentile=50.0,
        )

        sky_2d_e = bkg.background
        sky_rms_2d_e = bkg.background_rms

        data_clean_e = data_e - sky_2d_e

        if add_bkg_rms:
            # Only use this if self.sigma2 does NOT already include background noise
            sigma2_clean_e = sigma2_e + sky_rms_2d_e**2
        else:
            sigma2_clean_e = sigma2_e

        noise_clean_e = np.sqrt(sigma2_clean_e)

        # --------------------------------------------------
        # Diagnostics
        # --------------------------------------------------
        mean_resid = np.nanmean(data_clean_e[empty])
        median_resid = np.nanmedian(data_clean_e[empty])
        std_resid = np.nanstd(data_clean_e[empty])
        median_sky_rms = np.nanmedian(sky_rms_2d_e[empty])

        rms_ratio = std_resid / median_sky_rms

        z_bkg = data_clean_e[empty] / sky_rms_2d_e[empty]
        z_bkg_mean = np.nanmean(z_bkg)
        z_bkg_std = np.nanstd(z_bkg)

        z_noise = data_clean_e[empty] / noise_clean_e[empty]
        z_noise_mean = np.nanmean(z_noise)
        z_noise_std = np.nanstd(z_noise)

        offset_ok = np.abs(median_resid) < 0.05 * std_resid
        rms_ok = 0.8 < rms_ratio < 1.25
        z_ok = 0.8 < z_bkg_std < 1.25

        is_fine = offset_ok and rms_ok and z_ok

        if verbose:
            print("Background diagnostics")
            print("----------------------")
            print(f"mean_resid        = {mean_resid:.4g} e-")
            print(f"median_resid      = {median_resid:.4g} e-")
            print(f"std_resid         = {std_resid:.4g} e-")
            print(f"median sky RMS    = {median_sky_rms:.4g} e-")
            print(f"std / sky_rms     = {rms_ratio:.3f}")
            print(f"z_bkg mean/std    = {z_bkg_mean:.3f}, {z_bkg_std:.3f}")
            print(f"z_noise mean/std  = {z_noise_mean:.3f}, {z_noise_std:.3f}")

            if is_fine:
                print("OK: background subtraction looks fine.")
            else:
                print("WARNING: background subtraction may not be fine.")

                if not offset_ok:
                    print("- Residual background is not centered close enough to zero.")

                if not rms_ok:
                    print("- Empirical RMS and Background2D RMS differ significantly.")

                if not z_ok:
                    print("- Normalized residuals are not close to unit width.")

        if plot:
            nice_psf_plot(data_clean_e,figsize=(20, 20),colorlabel="e-",)

        # --------------------------------------------------
        # Optional: update object attributes to electrons
        # --------------------------------------------------
        if inplace:
            print("data and sigma2 will be replaced")
            self.data = data_clean_e
            self.sigma2 = sigma2_clean_e
            self.noise_map = noise_clean_e
            self.sky_2d = sky_2d_e
            self.sky_rms_2d = sky_rms_2d_e
            self.source_mask = source_mask
            self.units = "e-"

        #return data_clean_e, noise_clean_e, sigma2_clean_e, diagnostics



    def _detect_lens_(self,seplimit=0.001):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        coords_obj = SkyCoord(ra=self.RA * u.deg, dec=self.DEC * u.deg,frame="icrs",)
        coords_b = SkyCoord(ra=tabla1["RAdeg"].to_numpy() * u.deg,dec=tabla1["DEdeg"].to_numpy() * u.deg,frame="icrs",)

        sep2d = coords_obj.separation(coords_b)
        mask = sep2d < seplimit * u.deg

        matched_b = tabla1.loc[mask].reset_index(drop=False)
        matched_b["sep_arcsec"] = sep2d[mask].arcsec
        mask  = np.array(["G" in i for i in matched_b["Comp"].values])
        self.images_coordinates = matched_b[["Comp","RAdeg","DEdeg"]][~mask]
        self.galaxy_coordinates = matched_b[["Comp","RAdeg","DEdeg"]][mask]
        self.lens_table = matched_b
    
    def _detect_lens(self):

        ny, nx = self.data.shape

        wcs = self.wcs

        wcs = wcs.celestial

        coords_cat = SkyCoord(ra=tabla1["RAdeg"].to_numpy() * u.deg,dec=tabla1["DEdeg"].to_numpy() * u.deg, frame="icrs",)

       
        xpix, ypix = wcs.world_to_pixel(coords_cat)
        
        inside_image = (np.isfinite(xpix) & np.isfinite(ypix) & (xpix >= -0) & (xpix < nx + 0) & (ypix >= -0) & (ypix < ny + 0))


        sep2d = None
        mask = inside_image

        # ------------------------------------------------------------
        # 5. Store matched catalog
        # ------------------------------------------------------------
        matched_b = tabla1.loc[mask].copy().reset_index(drop=False)

        matched_b["xpix"] = xpix[mask]
        matched_b["ypix"] = ypix[mask]
        self.lens_table = matched_b
        if len(matched_b)==0:
            return matched_b
        size = u.Quantity([20, 20], u.pixel)#razonable?
        x,y = matched_b[["xpix","ypix"]].values.T
        x_center = (x.min() + x.max()) / 2
        y_center = (y.min() + y.max()) / 2
        cutout3 = Cutout2D(self.data.data, (x_center, y_center), size)
        
        percente_nan = np.sum(np.isnan(cutout3.data))/len(cutout3.data.ravel())
        #print(percente_nan)
        if percente_nan>0.3:
           self.lens_table = tabla1[tabla1["Name"]=="a"]
           return  self.lens_table

        if sep2d is not None:
            matched_b["sep_arcsec"] = sep2d[mask].arcsec

        is_galaxy = matched_b["Comp"].astype(str).str.contains("G", na=False)

        self.images_coordinates = matched_b.loc[~is_galaxy, ["Comp", "RAdeg", "DEdeg", "xpix", "ypix"]].reset_index(drop=True)

        self.galaxy_coordinates = matched_b.loc[is_galaxy, ["Comp", "RAdeg", "DEdeg", "xpix", "ypix"]].reset_index(drop=True)

        self.lens_table = matched_b

        return matched_b
    




    def cut_out_lens(self, times_maxsep=2, plot=False, refine_centers=True, 
                     centroid_box_size=15, centroid_method="2dg",
                     num_pix=51,detec_fwhm=None,threshold=None,
                     skycorr = np.array([0,0]),norm=LogNorm(1e-6)):
        """
        Create a cutout around the lens system and optionally refine the
        image positions using local centroiding.

        Parameters
        ----------
        times_maxsep : float, optional
            Multiplicative factor for the cutout size based on MaxSep.

        plot : bool, optional
            If True, show the cutout and image positions.

        refine_centers : bool, optional
            If True, refine the image positions using local centroiding.

        centroid_box_size : int, optional
            Size of the local box used for centroiding, in pixels.

        centroid_method : {"2dg", "com"}, optional
            Centroiding method. "2dg" uses a 2D Gaussian fit. "com" uses
            center of mass.
        """
        if detec_fwhm is None:
            detec_fwhm = self.default_fwhm
        if threshold is None:
            threshold = self.default_threshold
        
        self.num_pix = num_pix
        sky_pointing = SkyCoord(*self.galaxy_coordinates[["RAdeg", "DEdeg"]].values[0]+skycorr,unit="deg",)
        cutout_2d = Cutout2D(self.data, sky_pointing, num_pix ,wcs=self.wcs,)
        images_skycoord = SkyCoord(*self.images_coordinates[["RAdeg", "DEdeg"]].values.T, unit="deg",)

        coord_pix_initial = np.array(cutout_2d.wcs.world_to_pixel(images_skycoord)).T
        coord_pix_refined = coord_pix_initial.copy()
        print(f"values use for fwhm = {detec_fwhm} and threshold = {threshold}")
        daofind = DAOStarFinder(fwhm=detec_fwhm, threshold=threshold)  
        sources = daofind(cutout_2d.data)
        positions = np.transpose((sources["x_centroid"],sources["y_centroid"],))
        if refine_centers:
            x_init = coord_pix_initial[:, 0]
            y_init = coord_pix_initial[:, 1]

            if centroid_method == "2dg":
                centroid_func = centroid_2dg
            elif centroid_method == "com":
                centroid_func = centroid_com
            else:
                raise ValueError("centroid_method must be '2dg' or 'com'.")

            try:
                x_refined, y_refined = centroid_sources(cutout_2d.data, x_init, y_init, box_size=centroid_box_size, centroid_func=centroid_func,)
                coord_pix_refined = np.array([x_refined, y_refined]).T
                # If some centroiding failed and returned NaN, keep original values
                bad = ~np.isfinite(coord_pix_refined).all(axis=1)
                coord_pix_refined[bad] = coord_pix_initial[bad]

            except Exception as e:
                print(f"Centroid refinement failed: {e}")
                coord_pix_refined = coord_pix_initial.copy()
        dif = np.where(np.linalg.norm(positions[:, np.newaxis] - coord_pix_refined, axis=2) < 1)
        idx_remove = np.unique(dif[0])

        mask = np.ones(len(positions), dtype=bool)
        mask[idx_remove] = False
        positions = positions[mask]
        if plot:
            # fig = plt.figure(figsize=(20, 10))
            # axis1 = fig.add_subplot(1, 1, 1, projection=cutout_2d.wcs)
            fig,axis1  = plot_image_with_scalebar(cutout_2d.data,cutout_2d.wcs,cmap= "Greys",norm=norm)
            #axis1.imshow(cutout_2d.data,cmap="Greys",norm=LogNorm(1e-6), origin="lower",)

            for n, pt in enumerate(coord_pix_initial):
                axis1.scatter(pt[0],pt[1],c="red",marker="x",s=120)

            for n, pt in enumerate(coord_pix_refined):
                comp = self.images_coordinates[["Comp"]].values[n][0]

                axis1.scatter(pt[0],pt[1],c="white",marker="o",s=80,linewidths=2)
                axis1.text(pt[0],pt[1],comp,c="white",fontsize=20,)
            for n,p in enumerate(positions):
                axis1.text(p[0],p[1],"G?",c="orange",fontsize=20,)
                axis1.scatter(p[0],p[1],c="orange",marker="o",s=80,linewidths=2)
            plt.show()

        self.cutout_2d = cutout_2d
        if not hasattr(self, "sigma2"):
            from astropy.nddata import NDData
            cutouts_weights = Cutout2D(self.weights, sky_pointing, num_pix, wcs=self.wcs,)
            norm_factor,norm_data,norm_sigma2,data,sigma2 = normalize_data_error(cutout_2d.data,cutouts_weights,print_=False)#stars
            self.cutouts_sigma2 =  NDData(data=sigma2,wcs=self.wcs)
            #print("the cutout will run automatically")
            #self.cut_out_lens(plot=True)
        else:
            cutouts_sigma2 = Cutout2D(self.sigma2, sky_pointing, num_pix, wcs=self.wcs,)
            self.cutouts_sigma2 = cutouts_sigma2
            cutout = cutout_2d.data
            sigma_slice =np.sqrt(cutouts_sigma2.data)
            # Fraction of each side used as corner boxes
            ny, nx = cutout.shape

            corner_frac = 0.20
            dy = int(ny * corner_frac)
            dx = int(nx * corner_frac)

            corner_mask = np.zeros_like(cutout, dtype=bool)

            # True means: use this pixel for sky
            corner_mask[:dy, :dx] = True          # bottom/left depending on orientation
            corner_mask[:dy, -dx:] = True
            corner_mask[-dy:, :dx] = True
            corner_mask[-dy:, -dx:] = True

            bad = (
                ~np.isfinite(cutout)
                | ~np.isfinite(sigma_slice)
                | (sigma_slice <= 0)
            )

            # sigma_clipped_stats mask=True means "ignore this pixel"
            sky_mask = ~corner_mask | bad

            _, sky_median, sky_std = sigma_clipped_stats(
                cutout,
                mask=sky_mask,
                sigma=3.0,
                maxiters=10,
            )
            bad = ~np.isfinite(cutout) | ~np.isfinite(sigma_slice) | (sigma_slice <= 0)
            data_skysub = cutout - sky_median

            var = sigma_slice**2
            
            source_rate = np.clip(data_skysub, 0.0, None)
            poisson_var = source_rate / self.exptime_seconds
            var = var + poisson_var

            noise_corrected = np.sqrt(var)

            self.data_skysub = np.where(bad, np.nan, data_skysub)
            self.noise_corrected = np.where(bad, np.nan, noise_corrected)


        self.coord_pix_images_initial = {self.images_coordinates[["Comp"]].values[n][0]: pt for n, pt in enumerate(coord_pix_initial)}
        self.coord_pix_images = {self.images_coordinates[["Comp"]].values[n][0]: pt for n, pt in enumerate(coord_pix_refined)}
        self.coord_pix_images_refined = self.coord_pix_images
        self.coord_pix_non_images = positions

        mask_object_region = np.zeros(self.data.shape, dtype=bool)
        mask_object_region[self.cutout_2d.slices_original] = True
        mask_object_region |= ~np.isfinite(self.data)

        mask_object_region = binary_dilation(mask_object_region,iterations=20)

        self.mask_object_region = mask_object_region

    
    def detect_stars(self,detec_fwhm=None,threshold=None,verbose=True,make_plots=False,num_pix=None,n_keep=20,binary_dilation_iteration=20
                     ,use_gaia=True,gaia_gmag_limit=20,gaia_radius=None,refine_star_centers=True,star_centroid_box_size=7,star_centroid_method="2dg",
                     percent=99,roundness_range=(-1,0.5)):
        if not hasattr(self, "cutout_2d"):
            print("the cutout will run automatically")
            self.cut_out_lens(plot=True)
        if detec_fwhm is None:
            detec_fwhm = self.default_fwhm
        if num_pix is None:
            num_pix = self.num_pix
        if threshold is None:
            threshold = self.default_threshold
        
        if verbose:
            print(f"The search for stars will be with " f"detec_fwhm = {detec_fwhm:.3f} and "f"threshold = {threshold:.3f}")

        
        sources = None
        if use_gaia:
            try:
                print("Querying Gaia catalog for stars in the field...")
                from astroquery.vizier import Vizier
                ny, nx = self.data.shape

                # Use image center if gaia_radius is not provided
                if hasattr(self, "RA") and hasattr(self, "DEC"):
                    center = SkyCoord(self.RA * u.deg,self.DEC * u.deg,frame="icrs",)
                else:
                    center = self.wcs.pixel_to_world(nx / 2, ny / 2)

                # Estimate radius from WCS image corners
                if gaia_radius is None:
                    corners_x = np.array([0, nx - 1, 0, nx - 1])
                    corners_y = np.array([0, 0, ny - 1, ny - 1])

                    corners_world = self.wcs.pixel_to_world(corners_x, corners_y)
                    gaia_radius = center.separation(corners_world).max() + 5 * u.arcsec

                viz = Vizier(
                    columns=["Source","RA_ICRS","DE_ICRS","Gmag","BPmag","RPmag","Plx","pmRA","pmDE",],
                    column_filters={
                        "Gmag": f"<{gaia_gmag_limit}",
                    },
                    row_limit=-1,)

                result = viz.query_region(
                    center,
                    radius=gaia_radius,
                    catalog="I/355/gaiadr3",)
                print(f"Gaia query returned {len(result)} sources within {gaia_radius.to(u.arcsec)} of the image center.")
                if len(result) == 0:
                    if verbose:
                        print("No Gaia sources found in the field.")
                if len(result) > 0 and len(result[0]) > 0:
                    gaia = result[0]
                    gaia_coord = SkyCoord(gaia["RA_ICRS"],gaia["DE_ICRS"],unit="deg",frame="icrs",)
                    x_gaia, y_gaia = self.wcs.world_to_pixel(gaia_coord)

                    x_gaia = np.asarray(x_gaia)
                    y_gaia = np.asarray(y_gaia)

                    finite_xy = np.isfinite(x_gaia) & np.isfinite(y_gaia)

                    inside_image = ((x_gaia >= 0) & (x_gaia < nx) & (y_gaia >= 0) & (y_gaia < ny))

                    valid = finite_xy & inside_image
                    
                    x_int = np.rint(x_gaia[valid]).astype(int)
                    y_int = np.rint(y_gaia[valid]).astype(int)

                    outside_object_mask = ~self.mask_object_region[y_int, x_int]

                    valid_indices = np.where(valid)[0][outside_object_mask]
                    print(f"Found {len(valid_indices)} Gaia sources with valid positions outside the object region.")
                    if len(valid_indices) > 0:
                        sources = gaia[valid_indices]

                        # Add columns compatible with the rest of your code
                        sources["x_centroid"] = x_gaia[valid_indices]
                        sources["y_centroid"] = y_gaia[valid_indices]
                        positions = np.transpose((sources['x_centroid'], sources['y_centroid']))
                        # Artificial flux-like quantity for sorting by brightness.
                        # Brighter Gaia stars have smaller Gmag, so this increases with brightness.
                        sources["flux"] = 10 ** (-0.4 * np.asarray(sources["Gmag"]))

                        if len(sources) > n_keep:
                            sources.sort("flux")
                            sources.reverse()
                            sources = sources[:n_keep]

                        if verbose:
                            print(f"Using {len(sources)} Gaia stars in the field.")

            except Exception as e:
                if verbose:
                    print(f"Gaia query failed, falling back to DAOStarFinder. Reason: {e}")

        if sources is None or len(sources) == 0:
            daofind = DAOStarFinder(fwhm=detec_fwhm,threshold=threshold, roundness_range=roundness_range,exclude_border=False,)
            sources = daofind(self.data - self.median, mask=self.mask_object_region)
            positions = np.transpose((sources['x_centroid'], sources['y_centroid']))

            if sources is not None and len(sources) > n_keep:
                sources.sort("flux")
                sources.reverse()
                sources = sources[:n_keep]

        if sources is None or len(sources) == 0:
            if verbose:
                print("No stars were detected.")

            positions = np.empty((0, 2))
            apertures_stars = None
            cutouts_stars = []
            cutouts_weights = []
            ra_dec = np.empty((0, 2))

            return sources

        #positions_stars = np.transpose((sources['x_centroid'], sources['y_centroid']))
        apertures_stellar = CircularAperture(positions, r=5.0)
        apertures_annulus = CircularAnnulus(positions, r_in=9, r_out=12)
        apertures_stars = CircularAperture(positions, r=10.0)
        self.phot_table = aperture_photometry(self.data, apertures_stellar)
        if verbose:
            print(f"Detected {len(positions)} stars outside the object region.")

        if make_plots:
            image = self.data
            fig, ax = plt.subplots(figsize=(20, 20))
            norm = simple_norm(image, "sqrt", percent=percent)

            ax.imshow(image,cmap="gray",origin="lower",aspect="equal",interpolation="nearest",norm=norm,)

            # Mask must have the same shape as image
            mask = np.zeros(self.data.shape, dtype=bool)
            mask[self.cutout_2d.slices_original] = True

            contours = find_contours(mask.astype(float), level=0.5)

            if len(contours) > 0:
                # Select largest connected contour
                contours = max(contours, key=len)
                # skimage returns (y, x), convert to (x, y)
                vertices = np.column_stack([contours[:, 1], contours[:, 0]])
                patch = Polygon(vertices,closed=True,fill=False,edgecolor="blue",linewidth=2,)
                ax.add_patch(patch)

            apertures_stars.plot(ax=ax,color="red",lw=1.5,alpha=0.7,)

            for i, (x, y) in enumerate(positions):
                ax.text(x,y,str(i),color="red",fontsize=12,)
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(0, image.shape[0])

            plt.show()
        x,y = positions[:, 0],positions[:, 1]
        coord_world = self.wcs.pixel_to_world(x, y)
        self.ra_dec = np.column_stack([coord_world.ra.deg,coord_world.dec.deg,])
        self.num_pix = num_pix
        self.cutstars(num_pix,verbose=verbose,plot_stars=make_plots)
        
        return sources


    def from_csv(self,reg_path,percent=99,verbose=True,make_plots=True):
        if not hasattr(self, "cutout_2d"):
            print("the cutout will run automatically")   
        # if detec_fwhm is None:
        #     detec_fwhm = self.default_fwhm
        # if threshold is None:
        #     threshold = self.default_threshold
        reg_path = Path(reg_path)
        if not reg_path.is_file():
            raise FileNotFoundError(f"File not found: {reg_path}")
        positions = pd.read_csv(reg_path)[["X_IMAGE","Y_IMAGE"]].values
        #return positions
        apertures_stellar = CircularAperture(positions, r=5.0)
        apertures_annulus = CircularAnnulus(positions, r_in=9, r_out=12)
        apertures_stars = CircularAperture(positions, r=10.0)
        phot_table = aperture_photometry(self.data, apertures_stellar)
        if verbose:
            print(f"Detected {len(positions)} stars outside the object region.")

        if make_plots:
            image = self.data
            fig, ax = plt.subplots(figsize=(20, 20))
            norm = simple_norm(image, "sqrt", percent=percent)

            ax.imshow(image,cmap="gray",origin="lower",aspect="equal",interpolation="nearest",norm=norm,)

            # Mask must have the same shape as image
            mask = np.zeros(self.data.shape, dtype=bool)
            mask[self.cutout_2d.slices_original] = True

            contours = find_contours(mask.astype(float), level=0.5)

            if len(contours) > 0:
                # Select largest connected contour
                contours = max(contours, key=len)
                # skimage returns (y, x), convert to (x, y)
                vertices = np.column_stack([contours[:, 1], contours[:, 0]])
                patch = Polygon(vertices,closed=True,fill=False,edgecolor="blue",linewidth=2,)
                ax.add_patch(patch)

            apertures_stars.plot(ax=ax,color="red",lw=1.5,alpha=0.7,)

            for i, (x, y) in enumerate(positions):
                ax.text(x,y,str(i),color="red",fontsize=12,)
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(0, image.shape[0])

            plt.show()


    def from_region(self,reg_path,detec_fwhm=None,threshold=None,verbose=True,star_num_pix=51,binary_dilation_iteration=20
                    ,refine_star_centers=True,star_centroid_box_size=7,star_centroid_method="2dg",make_plots=True,percent=99.99,plot_stars=True):
        if not hasattr(self, "cutout_2d"):
            print("the cutout will run automatically")
            self.cut_out_lens(plot=True)
        if detec_fwhm is None:
            detec_fwhm = self.default_fwhm

        if threshold is None:
            threshold = self.default_threshold
        
        if verbose:
            print(f"The search for stars will be with " f"detec_fwhm = {detec_fwhm:.3f} and "f"threshold = {threshold:.3f}")
        
        reg_path = Path(reg_path)
        if not reg_path.is_file():
            raise FileNotFoundError(f"File not found: {reg_path}")
        
        wcs = self.wcs
        reg_to_sky_frame = pyregion.open(reg_path).as_imagecoord(header=self.header)
        cut_out_list = []
        star_pos_list = []
        for reg in reg_to_sky_frame:
            center_reg = reg.coord_list[:2]
            size = 40.0
            _,cut_out,star_pos = create_rectangle_patch(self.data,center_reg, size=size)
            if cut_out.shape != (size,size):
                continue
            cut_out_list.append(cut_out)
            star_pos_list.append(star_pos)
        star_pos_array = np.asarray(star_pos_list)
        daofind = DAOStarFinder(fwhm=detec_fwhm, threshold=threshold)  
        sources = daofind(self.data)
        positions = np.transpose((sources["x_centroid"],sources["y_centroid"],))
        #apertures_daostars = CircularAperture(positions, r=5.0)
        coords_dao = wcs.pixel_to_world(positions[:, 0],positions[:, 1],)
        
        coords_regions = wcs.pixel_to_world(star_pos_array[:, 0],star_pos_array[:, 1],)
        max_sep = 1.0 * u.arcsec
        idx, sep2d, _ = coords_regions.match_to_catalog_sky(coords_dao)
        good_match = sep2d < max_sep
        #positions_stars = positions[idx[good_match]]
        sources = sources[idx[good_match]]
        #apertures_stars = CircularAperture(positions_stars, r=10.0)
        
        positions_stars = np.transpose((sources["x_centroid"],sources["y_centroid"],))

        apertures_stars = CircularAperture(positions_stars, r=10.0)
        apertures_stellar = CircularAperture(positions_stars, r=5.0)
        self.phot_table = aperture_photometry(self.data, apertures_stellar)
        if verbose:
            print(f"Detected {len(positions_stars)} stars outside the object region.")
        
        if make_plots:
            image = self.data
            fig, ax = plt.subplots(figsize=(20, 20))
            norm = simple_norm(image, "sqrt", percent=percent)

            ax.imshow(image,cmap="gray",origin="lower",aspect="equal",interpolation="nearest",norm=norm,)

            # Mask must have the same shape as image
            mask = np.zeros(self.data.shape, dtype=bool)
            mask[self.cutout_2d.slices_original] = True

            contours = find_contours(mask.astype(float), level=0.5)

            if len(contours) > 0:
                # Select largest connected contour
                contours = max(contours, key=len)
                # skimage returns (y, x), convert to (x, y)
                vertices = np.column_stack([contours[:, 1], contours[:, 0]])
                patch = Polygon(vertices,closed=True,fill=False,edgecolor="blue",linewidth=2,)
                ax.add_patch(patch)

            apertures_stars.plot(ax=ax,color="red",lw=1.5,alpha=0.7,)

            for i, (x, y) in enumerate(positions_stars):
                ax.text(x,y,str(i),color="red",fontsize=12,)
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(0, image.shape[0])

            plt.show()
        #x,y = positions_stars[:, 0],positions_stars[:, 1]
        #coord_world = self.wcs.pixel_to_world(x, y)
        #self.recut_stars(star_num_pix)
        self.cutstars(star_num_pix,verbose=verbose,plot_stars=plot_stars)

        return sources

    # def recut_stars_(self,star_num_pix):
    #     self.stars2D_data = [Cutout2D(self.data, self.positions_stars[i], star_num_pix, wcs=self.wcs).data for i in range(len(self.positions_stars))]
    #     self.starst2D_sigma2 = [Cutout2D(self.sigma2, self.positions_stars[i], star_num_pix, wcs=self.wcs).data for i in range(len(self.positions_stars))]
    
    def cutstars(self,star_num_pix,verbose=True,plot_stars=True):
        star_ids = list(self.phot_table['id'].value)
        x,y = self.phot_table['x_center'].value,self.phot_table['x_center'].value
        coord_world = self.wcs.pixel_to_world(x, y)
        self.ra_dec = np.column_stack([coord_world.ra.deg,coord_world.dec.deg,])
        xis = [x+1 for x in list(self.phot_table['x_center'].value)]
        yis = [y+1 for y in list(self.phot_table['y_center'].value)]
        self.stars2D_data,self.starst2D_sigma2,positions_stars = make_cutouts(image=self.data,
                                    sigma2 = self.sigma2,
                                    star_ids=star_ids, 
                                    xis=xis, 
                                    yis=yis, 
                                    rpix=star_num_pix//2, 
                                    scale_stars=True, 
                                    show_figs=plot_stars, 
                                    sub_pixel=True,verbose=verbose)


    def save_for_starred(self,non_selected_stars_index=[],do_plots=False,verbose=False,save_path=None,star_num_pix=None,
                         plot_stars_percentile=(0.1,99.9)):
        if not hasattr(self, "cutout_2d"):
            raise FileNotFoundError(f"cutout_2d not calculated")
        if not hasattr(self,"stars2D_data"):
            raise FileNotFoundError(f"stars2D_data not calculated")
        if star_num_pix is not None and (isinstance(star_num_pix, (int, float))):
            self.cutstars(star_num_pix)
            

        n_stars = len(self.stars2D_data)
        selected_star_indices = [i for i in range(n_stars) if i not in non_selected_stars_index]

        data =  self.data_skysub[None,:]
        sigma2 = (self.noise_corrected**2)[None,:]
        # #stars_data = np.stack([self.stars2D_data[i].data for i in range(n_stars) if i not in non_selected_stars_index])
        # #stars_sigma2 = np.stack([self.starst2D_sigma2[i].data for i in range(n_stars) if i not in non_selected_stars_index])
        stars_data = np.stack([self.stars2D_data[i] for i in range(n_stars) if i not in non_selected_stars_index])
        stars_sigma2 = np.stack([self.starst2D_sigma2[i] for i in range(n_stars) if i not in non_selected_stars_index])
        radec = np.stack([self.ra_dec[i] for i in range(n_stars) if i not in non_selected_stars_index])
        coord_pix_images = list(self.coord_pix_images.values())
        images_names = list(self.coord_pix_images.keys())
        starred_dict = {"data": np.asarray(data),"sigma2":np.asarray(sigma2),"stars_data":stars_data,"stars_sigma2":stars_sigma2}
       
        starred_dict.update({"coord_pix_images":np.array(coord_pix_images), "images_names":images_names,"coord_pix_images_dict": self.coord_pix_images,
                            "selected_star_indices": selected_star_indices, 
                             "non_selected_stars_index": non_selected_stars_index,
                             "coord_pix_non_images":self.coord_pix_non_images,"pixel_scale":self.pixel_scale,"zp_ab": self.zp_ab,"exptime_seconds":self.exptime_seconds})
                        

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "wb") as f:
                pickle.dump(starred_dict, f)

            if verbose:
                print(f"Saved STARRED dictionary to: {save_path}")
            
        if do_plots:
            plot_stars_features(stars_data,stars_sigma2,radec=radec,index =selected_star_indices,percentile=plot_stars_percentile)

        return starred_dict
    
    def refine_star_centers(self,positions,box_size=15,centroid_method="2dg",verbose=True,):
        """
        Refine star centers using local centroiding.

        Parameters
        ----------
        positions : ndarray
            Initial star positions with shape (N, 2), in full-image pixel coordinates.

        box_size : int, optional
            Size of the local centroiding box in pixels.

        centroid_method : {"2dg", "com"}, optional
            Centroiding method. "2dg" uses a 2D Gaussian fit. "com" uses
            center of mass.

        verbose : bool, optional
            If True, print basic information.

        Returns
        -------
        refined_positions : ndarray
            Refined star positions with shape (N, 2).
        """
        from photutils.centroids import centroid_sources, centroid_2dg, centroid_com

        positions = np.asarray(positions, dtype=float)

        if len(positions) == 0:
            return positions

        if centroid_method == "2dg":
            centroid_func = centroid_2dg
        elif centroid_method == "com":
            centroid_func = centroid_com
        else:
            raise ValueError("centroid_method must be '2dg' or 'com'.")

        x_init = positions[:, 0]
        y_init = positions[:, 1]

        try:
            x_refined, y_refined = centroid_sources(
                self.data,
                x_init,
                y_init,
                box_size=box_size,
                centroid_func=centroid_func,
            )

            refined_positions = np.column_stack([x_refined, y_refined])

            bad = ~np.isfinite(refined_positions).all(axis=1)
            refined_positions[bad] = positions[bad]

            if verbose:
                n_bad = np.sum(bad)
                print(f"Refined {len(positions)} star centers. Failed: {n_bad}")

            return refined_positions

        except Exception as e:
            if verbose:
                print(f"Star centroid refinement failed: {e}")
                print("Using original detected positions.")

            return positions.copy()
    
    def plot_full(self,percent=99):
        image = self.data
        fig, ax = plt.subplots(figsize=(20, 20))
        norm = simple_norm(image, "sqrt", percent=percent)

        ax.imshow(image,cmap="gray",origin="lower",aspect="equal",interpolation="nearest",norm=norm,)

        plt.show()