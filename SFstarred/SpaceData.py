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
import pyregion

from scipy.ndimage import binary_dilation


from SFstarred.utils import find_nearest_object,create_rectangle_patch,read_jwst_for_starred
from SFstarred.plots import plot_image_with_scalebar,plot_stars_features
from SFstarred.stars_cut import normalize_data_error,make_sigma2_from_hst_weight


tabla1 = pd.read_csv(Path(__file__).resolve().parent / "suportdata" / "tablea1.csv")

class SpaceData:
    def __init__(self, path_data):
        path_file = Path(path_data)
        self.path_file = path_file
        if not path_file.is_file():
                raise FileNotFoundError(f"File not found: {path_file}")
        self._readfits()
        self.found_object()
        #filter object 
       
        
        
    def _readfits(self):
        self.header0 = fits.open(self.path_file)[0].header
        data, self.header = fits.getdata(self.path_file, header=True)
        self.TELESCOP = self.header0.get("TELESCOP")
        if  self.TELESCOP=="JWST":
            
            self.TARGNAME = self.header0.get("TARGPROP")
            self.RA = self.header0["TARG_RA"]
            self.DEC = self.header0["TARG_DEC"]
            self.FILTER  = self.header0.get("FILTER")
            #= self.header0.get("EFFEXPTM")
            self.data, self.sigma2, self.exptime_seconds = read_jwst_for_starred(self.path_file,add_poisson_from_sci=True)
            self.mean, self.median, self.std = sigma_clipped_stats(self.data, sigma=5.0)
        else:
            self.TARGNAME = self.header.get("TARGNAME")
            self.RA = self.header0["RA_TARG"]
            self.DEC = self.header0["DEC_TARG"]
            self.EXPTIME = self.header0.get("EXPTIME")
            self.FILTER  = self.header0.get("FILTER")
            self.data = data.astype(float)
            self.mean, self.median, self.std = sigma_clipped_stats(self.data, sigma=5.0)
            #self.data = data - self.median
        self.wcs = WCS(self.header)
        self.default_fwhm = 5
        self.default_threshold = 50.0 * self.std
        if self.FILTER == "F160W":
            self.default_fwhm = 6
            self.default_threshold = 50.0 * self.std
        elif self.FILTER == "F814W":
            self.default_fwhm = 5
            self.default_threshold = 10.0 * self.std

        elif self.FILTER == "F475X":
            self.default_fwhm = 5
            self.default_threshold = 10.0 * self.std
        
    
    def found_object(self,seplimit=0.001):
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
    def cut_light(self,plot=True,num_pix=51,detec_fwhm=None,threshold=None
                  ,percentiles=(1,99),add_colorbar=True,norm=LogNorm(1e-6),cmap="Greys",
                  refine_centers=True,centroid_method="2dg",centroid_box_size=2):
        if detec_fwhm is None:
            detec_fwhm = self.default_fwhm
        if threshold is None:
            threshold = self.default_threshold
        self.num_pix = num_pix
        sky_pointing = SkyCoord(*self.galaxy_coordinates[["RAdeg", "DEdeg"]].values[0],unit="deg")
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
        
        if plot:
            # fig = plt.figure(figsize=(20, 10))
            # axis1 = fig.add_subplot(1, 1, 1, projection=cutout_2d.wcs)
            fig,axis1  = plot_image_with_scalebar(cutout_2d.data,cutout_2d.wcs,
                                                  cmap= cmap,norm=norm,
                                                  filter=self.FILTER,percentiles=percentiles,add_colorbar=add_colorbar)
            for n, pt in enumerate(coord_pix_refined):
                comp = self.images_coordinates[["Comp"]].values[n][0]

                axis1.scatter(pt[0],pt[1],c="b",marker="o",s=80,linewidths=2)
                axis1.text(pt[0],pt[1],comp,c="b",fontsize=20,)
            
            plt.show()

        self.cutout_2d = cutout_2d
    
      
    def cut_out_lens(self, times_maxsep=2, plot=False, refine_centers=True, centroid_box_size=15, centroid_method="2dg",num_pix=51,detec_fwhm=None,threshold=None):
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
        sky_pointing = SkyCoord(*self.galaxy_coordinates[["RAdeg", "DEdeg"]].values[0],unit="deg",)
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
        dif = np.where(np.linalg.norm(positions[:, np.newaxis] - coord_pix_refined, axis=2) < 1e-1)
        idx_remove = np.unique(dif[0])

        mask = np.ones(len(positions), dtype=bool)
        mask[idx_remove] = False
        positions = positions[mask]
        if plot:
            # fig = plt.figure(figsize=(20, 10))
            # axis1 = fig.add_subplot(1, 1, 1, projection=cutout_2d.wcs)
            fig,axis1  = plot_image_with_scalebar(cutout_2d.data,cutout_2d.wcs,cmap= "Greys",norm=LogNorm(1e-6),filter=self.FILTER)
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
        self.coord_pix_images_initial = {self.images_coordinates[["Comp"]].values[n][0]: pt for n, pt in enumerate(coord_pix_initial)}
        self.coord_pix_images = {self.images_coordinates[["Comp"]].values[n][0]: pt for n, pt in enumerate(coord_pix_refined)}
        self.coord_pix_images_refined = self.coord_pix_images
        self.coord_pix_non_images = positions
