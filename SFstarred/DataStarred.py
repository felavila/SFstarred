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


from SFstarred.utils import find_nearest_object,create_rectangle_patch

tabla1 = pd.read_csv(Path(__file__).resolve().parent / "suportdata" / "tablea1.csv")

class DataStarred:
    
    def __init__(self, path_data,path_weight):
        path_file = Path(path_data)
        path_weight = Path(path_weight)
        
        if not path_file.is_file():
            raise FileNotFoundError(f"File not found: {path_file}")
        if not path_weight.is_file():
            raise FileNotFoundError(f"File weight not found: {path_weight}")

        self.path_file = path_file
        self.path_weight = path_weight
        self._readfits()
        self._readweight()
        self.found_object()
        #filter object 
       
        
        
    def _readfits(self):
        data_raw, self.header = fits.getdata(self.path_file, header=True)
        self.FILTER  = self.header.get("FILTER")
        self.TARGNAME = self.header.get("TARGNAME")
        self.RA = self.header.get("RA_TARG")
        self.DEC = self.header.get("DEC_TARG")
        self.EXPTIME = self.header.get("EXPTIME")
        self.wcs = WCS(self.header)
        data_raw = data_raw.astype(float)
        self.mean, self.median, self.std = sigma_clipped_stats(data_raw, sigma=5.0)
        self.data_raw = data_raw - self.median
        
    def _readweight(self):
        weights,_ = fits.getdata(self.path_weight, header=True)
        weights = weights.astype(float)
        self.header_weights = _
        RE_NORMALIZE_WEIGHTS = True
        #weights[weights==0] = np.NaN
        weights[weights==0] = np.nan
        wht_mean = weights[weights>0].mean()
        wht_max = weights[weights>0].max()
        wht_std = weights[weights>0].std()
        weights = weights / wht_max * self.EXPTIME
        self.weights = weights
    
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
        self.matched_b = matched_b
        
    def cut_out_lens(self, times_maxsep=2, plot=False, refine_centers=True, centroid_box_size=15, centroid_method="2dg",num_pix=51):
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

        cut_size = self.matched_b["MaxSep"].values[0] * times_maxsep
        self.num_pix = num_pix
        sky_pointing = SkyCoord(*self.galaxy_coordinates[["RAdeg", "DEdeg"]].values[0],unit="deg",)

        height_cut = cut_size * u.arcsec
        width_cut = cut_size * u.arcsec

        cutout_2d = Cutout2D(self.data_raw, sky_pointing, num_pix ,wcs=self.wcs,)

        cutouts_weight = Cutout2D(self.weights, sky_pointing, num_pix, wcs=self.wcs,)

        images_skycoord = SkyCoord(*self.images_coordinates[["RAdeg", "DEdeg"]].values.T, unit="deg",)

        coord_pix_initial = np.array(cutout_2d.wcs.world_to_pixel(images_skycoord)).T

        coord_pix_refined = coord_pix_initial.copy()

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
            fig = plt.figure(figsize=(20, 10))
            axis1 = fig.add_subplot(1, 1, 1, projection=cutout_2d.wcs)

            axis1.imshow(cutout_2d.data,cmap="Greys",norm=LogNorm(1e-6), origin="lower",)

            for n, pt in enumerate(coord_pix_initial):
                axis1.scatter(pt[0],pt[1],c="red",marker="x",s=120)

            for n, pt in enumerate(coord_pix_refined):
                comp = self.images_coordinates[["Comp"]].values[n][0]

                axis1.scatter(pt[0],pt[1],c="white",marker="o",s=80,facecolors="none",linewidths=2)
                axis1.text(pt[0],pt[1],comp,c="white",fontsize=20,)
            #axis1.legend()
            plt.show()

        self.cutout_2d = cutout_2d
        self.cutouts_weight = cutouts_weight

        self.coord_pix_images_initial = {self.images_coordinates[["Comp"]].values[n][0]: pt for n, pt in enumerate(coord_pix_initial)
        }

        self.coord_pix_images = {self.images_coordinates[["Comp"]].values[n][0]: pt for n, pt in enumerate(coord_pix_refined)}

        self.coord_pix_images_refined = self.coord_pix_images

        # # Optional: also store refined sky coordinates
        # refined_sky = cutout_2d.wcs.pixel_to_world(
        #     coord_pix_refined[:, 0],
        #     coord_pix_refined[:, 1],
        # )

        # self.images_coordinates_refined = self.images_coordinates.copy()
        # self.images_coordinates_refined["RAdeg_refined"] = refined_sky.ra.deg
        # self.images_coordinates_refined["DEdeg_refined"] = refined_sky.dec.deg
     
    def detect_stars(self,detec_fwhm=None,threshold=None,verbose=True,make_plots=False,save=False,path_filter=None,star_num_pix=51,n_keep=20,binary_dilation_iteration=20
                     ,use_gaia=True,gaia_gmag_limit=20,gaia_radius=None,):
        if not hasattr(self, "cutout_2d"):
            print("the cutout will run automatically")
            self.cut_out_lens(plot=True)

        if self.FILTER == "F160W":
            default_fwhm = 5
            default_threshold = 50.0 * self.std

        elif self.FILTER == "F814W":
            default_fwhm = 5
            default_threshold = 10.0 * self.std

        elif self.FILTER == "F475X":
            default_fwhm = 5
            default_threshold = 10.0 * self.std

        else:
            raise NotImplementedError(
                f"No detection defaults defined for filter {self.FILTER}"
            )

        if detec_fwhm is None:
            detec_fwhm = default_fwhm

        if threshold is None:
            threshold = default_threshold
        
        if verbose:
            print(
                f"The search for stars will be with "
                f"detec_fwhm = {detec_fwhm:.3f} and "
                f"threshold = {threshold:.3f}"
            )

        mask_object_region = np.zeros(self.data_raw.shape, dtype=bool)
        mask_object_region[self.cutout_2d.slices_original] = True
        mask_object_region |= ~np.isfinite(self.data_raw)

        mask_object_region = binary_dilation(
            mask_object_region,
            iterations=binary_dilation_iteration,
        )

        self.mask_object_region = mask_object_region

        sources = None

        if use_gaia:
            try:
                from astroquery.vizier import Vizier

                ny, nx = self.data_raw.shape

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

                if len(result) > 0 and len(result[0]) > 0:
                    gaia = result[0]
                    gaia_coord = SkyCoord(gaia["RA_ICRS"],gaia["DE_ICRS"],unit="deg",frame="icrs",)

                    x_gaia, y_gaia = self.wcs.world_to_pixel(gaia_coord)

                    x_gaia = np.asarray(x_gaia)
                    y_gaia = np.asarray(y_gaia)

                    finite_xy = np.isfinite(x_gaia) & np.isfinite(y_gaia)

                    inside_image = (
                        (x_gaia >= 0)
                        & (x_gaia < nx)
                        & (y_gaia >= 0)
                        & (y_gaia < ny)
                    )

                    valid = finite_xy & inside_image

                    # Check Gaia positions against the object/lens/NaN mask
                    x_int = np.rint(x_gaia[valid]).astype(int)
                    y_int = np.rint(y_gaia[valid]).astype(int)

                    outside_object_mask = ~mask_object_region[y_int, x_int]

                    valid_indices = np.where(valid)[0][outside_object_mask]

                    if len(valid_indices) > 0:
                        sources = gaia[valid_indices]

                        # Add columns compatible with the rest of your code
                        sources["x_centroid"] = x_gaia[valid_indices]
                        sources["y_centroid"] = y_gaia[valid_indices]

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
            daofind = DAOStarFinder(
                fwhm=detec_fwhm,
                threshold=threshold,
                sharpness_range=(0.2, 0.9),
                roundness_range=(-0.4, 0.4),
                exclude_border=True,
            )

            sources = daofind(self.data_raw, mask=mask_object_region)

            if sources is not None and len(sources) > n_keep:
                sources.sort("flux")
                sources.reverse()
                sources = sources[:n_keep]

        if sources is None or len(sources) == 0:
            if verbose:
                print("No stars were detected.")

            self.positions_stars = np.empty((0, 2))
            self.apertures_stars = None
            self.cutouts_stars = []
            self.cutouts_weights = []
            self.ra_dec = np.empty((0, 2))

            return sources

        self.sources = sources

        self.positions_stars = np.transpose((
                sources["x_centroid"],
                sources["y_centroid"],))

        self.apertures_stars = CircularAperture(self.positions_stars, r=10.0)

        if verbose:
            print(f"Detected {len(self.positions_stars)} stars outside the object region.")

        if make_plots:
            plt.figure(figsize=(20, 20))
            plt.imshow(self.data_raw, cmap="Greys", norm=LogNorm(1e-6))
            self.apertures_stars.plot(color="red", lw=1.5, alpha=0.7)

            for i, (x, y) in enumerate(self.positions_stars, start=0):
                plt.text(x, y, str(i), color="red", fontsize=12)

            plt.show()

        self.cutouts_stars = [Cutout2D(self.data_raw, tuple(pos), star_num_pix, wcs=self.wcs) for pos in self.positions_stars]

        x = self.positions_stars[:, 0]
        y = self.positions_stars[:, 1]

        coord_world = self.wcs.pixel_to_world(x, y)

        self.ra_dec = np.column_stack([coord_world.ra.deg,coord_world.dec.deg,])

        self.cutouts_weights = [Cutout2D(self.weights, tuple(pos), star_num_pix, wcs=self.wcs) for pos in self.positions_stars]

        return sources


    def from_region(self,reg_path,detec_fwhm=None,threshold=None,verbose=True,make_plots=False,save=False,path_filter=None,star_num_pix = 51):
        if not hasattr(self, "cutout_2d"):
            print("the cutout will run automatically")
            self.cut_out_lens(plot=True)
        
        if self.FILTER == "F160W":
            default_fwhm = 5
            default_threshold = 50.0 * self.std

        elif self.FILTER == "F814W":
            default_fwhm = 5
            default_threshold = 10.0 * self.std

        elif self.FILTER == "F475X":
            default_fwhm = 5
            default_threshold = 10.0 * self.std

        else:
            raise NotImplementedError(f"No detection defaults defined for filter {self.FILTER}")

        if detec_fwhm is None:
            detec_fwhm = default_fwhm

        if threshold is None:
            threshold = default_threshold

        if verbose:
            print(
                f"The search for stars will be with "
                f"detec_fwhm = {detec_fwhm:.3f} and "
                f"threshold = {threshold:.3f}"
            )
        
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
            _,cut_out,star_pos = create_rectangle_patch(self.data_raw,center_reg, size=size)
            if cut_out.shape != (size,size):
                continue
            cut_out_list.append(cut_out)
            star_pos_list.append(star_pos)
        star_pos_array = np.asarray(star_pos_list)
        
        daofind = DAOStarFinder(fwhm=detec_fwhm, threshold=threshold)  
        sources = daofind(self.data_raw)
        positions = np.transpose((sources["x_centroid"],sources["y_centroid"],))
        #apertures_daostars = CircularAperture(positions, r=5.0)
        coords_dao = wcs.pixel_to_world(positions[:, 0],positions[:, 1],)
        
        coords_regions = wcs.pixel_to_world(star_pos_array[:, 0],star_pos_array[:, 1],)
        max_sep = 1.0 * u.arcsec
        idx, sep2d, _ = coords_regions.match_to_catalog_sky(coords_dao)
        good_match = sep2d < max_sep
        positions_stars = positions[idx[good_match]]

        apertures_stars = CircularAperture(positions_stars, r=10.0)
    
        if make_plots:
            plt.figure(figsize=(20, 20))
            plt.imshow(self.data_raw, cmap="Greys", norm=LogNorm(1e-6))
            apertures_stars.plot(color="red", lw=1.5, alpha=0.7)

            for i, (x, y) in enumerate(positions_stars, start=0):
                plt.text(x, y, str(i), color="red", fontsize=12)

            plt.show()
        
        self.cutouts_stars = [Cutout2D(self.data_raw, positions_stars[i], star_num_pix, wcs=wcs) for i in range(len(positions_stars))]
        self.cutouts_weights = [Cutout2D(self.weights, positions_stars[i], star_num_pix, wcs=wcs) for i in range(len(positions_stars))]
        x = positions_stars[:, 0]
        y = positions_stars[:, 1]

        coord_world = self.wcs.pixel_to_world(x, y)

        self.ra_dec = np.column_stack([coord_world.ra.deg,coord_world.dec.deg,])
        
        #return coords_dao,coords_regions
    
    
    
    def save_for_starred(self,non_selected_stars_index=[],do_plots=False,verbose=False,save_path=None):
        from SFstarred.stars_cut import normalize_data_error
        if not hasattr(self, "cutout_2d"):
            raise FileNotFoundError(f"cutout_2d not calculated")
        if not hasattr(self,"cutouts_stars"):
            raise FileNotFoundError(f"cutouts_stars not calculated")
        n_stars = len(self.cutouts_stars)
        stars_cutout = np.stack([self.cutouts_stars[i].data for i in range(n_stars) if i not in non_selected_stars_index])
        stars_exp_map = np.stack([self.cutouts_weights[i].data for i in range(n_stars) if i not in non_selected_stars_index])
        selected_star_indices = [i for i in range(n_stars) if i not in non_selected_stars_index] 
        #norm,data_cutout,sigma2,data_cutout_copy,sigma2_copy
        norm_factor,stars_data_norm,stars_sigma2_norm,stars_data_cutout,stars_sigma2 = normalize_data_error(stars_cutout,stars_exp_map,print_=verbose)#stars
        cutout_2d =  self.cutout_2d.data[None,:]
        cutouts_weight = self.cutouts_weight.data[None,:]
        noisemaps = 1/ np.sqrt(cutouts_weight)
        stars = stars_data_norm
        noisemaps_stars = stars_sigma2_norm
        coord_pix_images = list(self.coord_pix_images.values())
        images_names = list(self.coord_pix_images.keys())
        
        starred_dict = {"data": np.asarray(cutout_2d),"data_weight": np.asarray(cutouts_weight), "data_noisemaps": np.asarray(noisemaps), "stars": np.asarray(stars), "noisemaps_stars": np.asarray(noisemaps_stars), "coord_pix_images": np.asarray(coord_pix_images),
        "images_names": images_names, "coord_pix_images_dict": self.coord_pix_images, "selected_star_indices": selected_star_indices, "non_selected_stars_index": non_selected_stars_index, "norm_factor": norm_factor,}

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "wb") as f:
                pickle.dump(starred_dict, f)

            if verbose:
                print(f"Saved STARRED dictionary to: {save_path}")
            
        if do_plots:
            for i in range(len(stars_cutout)):
                fig, axes = plt.subplots(1, 2, figsize=(8, 3))
                ax = axes[0]
                print("ra, dec",self.ra_dec[selected_star_indices[i]])
                ax.set_title(f"Star cutout {selected_star_indices[i]}")
                im = ax.imshow(stars_cutout[i], norm=LogNorm(1e-3, 1e4), cmap='gray_r')
                fig.colorbar(im, ax=ax)
                ax = axes[1]
                ax.set_title("Exposure map")
                im = plt.imshow(stars_exp_map[i])
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                plt.show()
            plt.figure(figsize=(5,5))
            plt.imshow(cutout_2d[0], cmap='gray')
            for n,i in enumerate(coord_pix_images):
                plt.text(*i,images_names[n],c="white",fontsize=20)
                plt.scatter(*i,c="red",s=100)
            plt.axis('off')
            plt.tight_layout(); plt.show()
        return starred_dict
    
    
    def plot_stars(self):
        cutouts_stars = self.cutouts_stars
        cutouts_weights=self.cutouts_weights
        for i in range(len(cutouts_stars)):
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            ax = axes[0]
            print("ra,dec",self.ra_dec[i])
            ax.set_title(f"Star cutout {i}")
            im = ax.imshow(cutouts_stars[i].data, norm=LogNorm(1e-3, 1e4), cmap='gray_r')
            fig.colorbar(im, ax=ax)
            ax = axes[1]
            ax.set_title("Exposure map")
            im = plt.imshow(cutouts_weights[i].data)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            # if save:
            #     plt.savefig(os.path.join(path_filter,f"star_cutout_{i+1}_{sky_coord.to_string(style='hmsdms', precision=2)}_{FILTER}.jpg"))
            plt.show()