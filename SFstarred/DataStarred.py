from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder, find_peaks
from photutils.aperture import CircularAperture
import matplotlib.cm as cm
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm, Normalize, TwoSlopeNorm

import pandas as pd 
from SFstarred.utils import find_nearest_object

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
    
    def cut_out_lens(self,times_maxsep=2,plot=False):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from astropy.nddata import Cutout2D
        
        cut_size = self.matched_b["MaxSep"].values[0]*times_maxsep
        sky_pointing = SkyCoord(*self.galaxy_coordinates[["RAdeg","DEdeg"]].values[0], unit='deg')
        height_cut,width_cut = cut_size * u.arcsec, cut_size* u.arcsec
        cutout_2d = Cutout2D(self.data_raw,sky_pointing,size=(height_cut,width_cut),wcs=self.wcs)
        coord_pix = np.array(cutout_2d.wcs.world_to_pixel(SkyCoord(*self.images_coordinates[["RAdeg","DEdeg"]].values.T, unit='deg'))).T
        fig = plt.figure(figsize=(20, 10))
        axis1 = fig.add_subplot(1, 1, 1, projection=cutout_2d.wcs)
        plt.imshow(cutout_2d.data, cmap='Greys', norm=LogNorm(1e-6))
        if plot:
            for n,pt in enumerate(coord_pix):
                plt.scatter(*pt,c="white")
                plt.text(*pt,self.images_coordinates[["Comp"]].values[n][0],c="white",fontsize=20)
            plt.show()
        self.cutout_2d = cutout_2d
        
    def plot_stars(self):
        cutouts_stars = self.cutouts_stars
        cutouts_weights=self.cutouts_weights
        for i in range(len(cutouts_stars)):
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            ax = axes[0]
            print(self.ra_dec[i])
            ax.set_title(f"Star cutout {i+1}")
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
            
    def detect_stars(self,detec_fwhm=None,threshold=None,verbose=True,make_plots=False,save=False,path_filter=None,star_num_pix=51,n_keep=20,binary_dilation_iteration=20
                     ,use_gaia=True,gaia_gmag_limit=20,gaia_radius=None,):
        from scipy.ndimage import binary_dilation
        from astropy.nddata import Cutout2D
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        import numpy as np

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

        # ---------------------------------------------------------
        # Mask lens/object region + NaNs
        # ---------------------------------------------------------
        mask_object_region = np.zeros(self.data_raw.shape, dtype=bool)
        mask_object_region[self.cutout_2d.slices_original] = True
        mask_object_region |= ~np.isfinite(self.data_raw)

        mask_object_region = binary_dilation(
            mask_object_region,
            iterations=binary_dilation_iteration,
        )

        self.mask_object_region = mask_object_region

        sources = None

        # ---------------------------------------------------------
        # 1. Gaia-first search
        # ---------------------------------------------------------
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

        self.positions_stars = np.transpose(
            (
                sources["x_centroid"],
                sources["y_centroid"],
            )
        )

        self.apertures_stars = CircularAperture(self.positions_stars, r=10.0)

        if verbose:
            print(f"Detected {len(self.positions_stars)} stars outside the object region.")

        if make_plots:
            plt.figure(figsize=(20, 20))
            plt.imshow(self.data_raw, cmap="Greys", norm=LogNorm(1e-6))
            self.apertures_stars.plot(color="red", lw=1.5, alpha=0.7)

            for i, (x, y) in enumerate(self.positions_stars, start=1):
                plt.text(x, y, str(i), color="red", fontsize=12)

            plt.show()

        self.cutouts_stars = [
            Cutout2D(self.data_raw, tuple(pos), star_num_pix, wcs=self.wcs)
            for pos in self.positions_stars
        ]

        x = self.positions_stars[:, 0]
        y = self.positions_stars[:, 1]

        coord_world = self.wcs.pixel_to_world(x, y)

        self.ra_dec = np.column_stack(
            [
                coord_world.ra.deg,
                coord_world.dec.deg,
            ]
        )

        self.cutouts_weights = [
            Cutout2D(self.weights, tuple(pos), star_num_pix, wcs=self.wcs)
            for pos in self.positions_stars
        ]

        return sources


    def _save_to_starred(self):
        return
# def get_gaia_stars(self,radius=None,mag_limit=20,verbose=True,make_plots=False,save=False,path_filter=None,star_num_pix=51,):
#         """
#         Query Gaia stars around the field and create cutouts.

#         Parameters
#         ----------
#         radius : astropy.units.Quantity, optional
#             Search radius around the object. If None, it is estimated from the image size.
#         mag_limit : float, optional
#             Maximum Gaia G magnitude. Lower values mean brighter stars.
#         verbose : bool, optional
#             Print information.
#         make_plots : bool, optional
#             Plot detected Gaia stars.
#         save : bool, optional
#             Save the plot.
#         path_filter : str, optional
#             Output directory.
#         star_num_pix : int, optional
#             Size of the star cutouts in pixels.

#         Returns
#         -------
#         gaia_table : astropy.table.Table
#             Gaia sources in the field after masking the lens/object region.
#         """

#         import os
#         import numpy as np
#         import matplotlib.pyplot as plt

#         import astropy.units as u
#         from astropy.coordinates import SkyCoord
#         from astropy.nddata import Cutout2D
#         from astroquery.gaia import Gaia
#         from photutils.aperture import CircularAperture
#         from matplotlib.colors import LogNorm

#         if not hasattr(self, "cutout_2d"):
#             print("The lens cutout will run automatically.")
#             self.cut_out_lens(plot=True)

#         # --------------------------------------------------
#         # Object center
#         # --------------------------------------------------
#         center_coord = SkyCoord(
#             self.galaxy_coordinates["RAdeg"].values[0] * u.deg,
#             self.galaxy_coordinates["DEdeg"].values[0] * u.deg,
#             frame="icrs",
#         )

#         # --------------------------------------------------
#         # Estimate field radius if not provided
#         # --------------------------------------------------
#         if radius is None:
#             ny, nx = self.data_raw.shape

#             corners_pix = np.array(
#                 [
#                     [0, 0],
#                     [0, ny - 1],
#                     [nx - 1, 0],
#                     [nx - 1, ny - 1],
#                 ]
#             )

#             corners_sky = self.wcs.pixel_to_world(
#                 corners_pix[:, 0],
#                 corners_pix[:, 1],
#             )

#             radius = np.max(center_coord.separation(corners_sky))

#         if verbose:
#             print(f"Searching Gaia stars within radius = {radius.to(u.arcmin):.3f}")
#             print(f"Using Gaia magnitude limit: G < {mag_limit}")

#         # --------------------------------------------------
#         # Gaia query
#         # --------------------------------------------------
#         query = f"""
#         SELECT
#             source_id,
#             ra,
#             dec,
#             phot_g_mean_mag,
#             phot_bp_mean_mag,
#             phot_rp_mean_mag,
#             parallax,
#             pmra,
#             pmdec
#         FROM gaiadr3.gaia_source
#         WHERE
#             1 = CONTAINS(
#                 POINT('ICRS', ra, dec),
#                 CIRCLE('ICRS', {center_coord.ra.deg}, {center_coord.dec.deg}, {radius.to(u.deg).value})
#             )
#             AND phot_g_mean_mag < {mag_limit}
#         """

#         job = Gaia.launch_job_async(query)
#         gaia_table = job.get_results()

#         if len(gaia_table) == 0:
#             if verbose:
#                 print("No Gaia stars found in the field.")

#             self.gaia_sources = gaia_table
#             self.gaia_positions_stars = np.empty((0, 2))
#             self.gaia_apertures_stars = None
#             self.gaia_cutouts_stars = []
#             self.gaia_cutouts_weights = []

#             return gaia_table

#         # --------------------------------------------------
#         # Convert Gaia RA/DEC to pixel coordinates
#         # --------------------------------------------------
#         gaia_coords = SkyCoord(
#             gaia_table["ra"] * u.deg,
#             gaia_table["dec"] * u.deg,
#             frame="icrs",
#         )

#         x_pix, y_pix = self.wcs.world_to_pixel(gaia_coords)

#         positions = np.transpose((x_pix, y_pix))

#         # --------------------------------------------------
#         # Keep only Gaia stars inside image boundaries
#         # --------------------------------------------------
#         ny, nx = self.data_raw.shape

#         inside_image = (
#             (x_pix >= 0)
#             & (x_pix < nx)
#             & (y_pix >= 0)
#             & (y_pix < ny)
#         )

#         positions = positions[inside_image]
#         gaia_table = gaia_table[inside_image]

#         # --------------------------------------------------
#         # Remove Gaia stars inside the lens/object cutout
#         # --------------------------------------------------
#         mask_object_region = np.zeros(self.data_raw.shape, dtype=bool)
#         mask_object_region[self.cutout_2d.slices_original] = True
#         mask_object_region |= ~np.isfinite(self.data_raw)

#         x_int = np.round(positions[:, 0]).astype(int)
#         y_int = np.round(positions[:, 1]).astype(int)

#         outside_object = ~mask_object_region[y_int, x_int]

#         positions = positions[outside_object]
#         gaia_table = gaia_table[outside_object]

#         if len(gaia_table) == 0:
#             if verbose:
#                 print("Gaia sources were found, but none remained outside the object region.")

#             self.gaia_sources = gaia_table
#             self.gaia_positions_stars = np.empty((0, 2))
#             self.gaia_apertures_stars = None
#             self.gaia_cutouts_stars = []
#             self.gaia_cutouts_weights = []
#             self.gaia_mask_object_region = mask_object_region

#             return gaia_table

#         # --------------------------------------------------
#         # Save outputs in the object
#         # --------------------------------------------------
#         self.gaia_sources = gaia_table
#         self.gaia_positions_stars = positions
#         self.gaia_apertures_stars = CircularAperture(positions, r=10.0)
#         self.gaia_mask_object_region = mask_object_region

#         if verbose:
#             print(f"Found {len(self.gaia_positions_stars)} Gaia stars outside the object region.")

#         # --------------------------------------------------
#         # Optional plot
#         # --------------------------------------------------
#         if make_plots:
#             plt.figure(figsize=(20, 20))
#             plt.imshow(self.data_raw, cmap="Greys", norm=LogNorm(1e-6))

#             self.gaia_apertures_stars.plot(
#                 color="red",
#                 lw=1.5,
#                 alpha=0.7,
#             )

#             for i, (x, y) in enumerate(self.gaia_positions_stars, start=1):
#                 mag = self.gaia_sources["phot_g_mean_mag"][i - 1]
#                 plt.text(
#                     x,
#                     y,
#                     f"{i} | G={mag:.1f}",
#                     color="red",
#                     fontsize=12,
#                 )

#             if save:
#                 if path_filter is None:
#                     path_filter = "."
#                 os.makedirs(path_filter, exist_ok=True)
#                 plt.savefig(
#                     os.path.join(path_filter, f"fits_with_gaia_stars_{self.FILTER}.pdf"),
#                     bbox_inches="tight",
#                 )

#             plt.show()

#         # --------------------------------------------------
#         # Build cutouts
#         # --------------------------------------------------
#         self.gaia_cutouts_stars = [
#             Cutout2D(
#                 self.data_raw,
#                 tuple(pos),
#                 star_num_pix,
#                 wcs=self.wcs,
#                 mode="partial",
#                 fill_value=np.nan,
#             )
#             for pos in self.gaia_positions_stars
#         ]

#         self.gaia_cutouts_weights = [
#             Cutout2D(
#                 self.weights,
#                 tuple(pos),
#                 star_num_pix,
#                 wcs=self.wcs,
#                 mode="partial",
#                 fill_value=np.nan,
#             )
#             for pos in self.gaia_positions_stars
#         ]

#         return gaia_table