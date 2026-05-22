import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    from SFstarred.DataStarred import DataStarred
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm, LogNorm, Normalize, TwoSlopeNorm
    import numpy as np
    from copy import deepcopy
    from starred.procedures.deconvolution_routines import multi_steps_deconvolution
    import marimo as mo

    return (DataStarred,)


@app.cell
def _(DataStarred):
    path_data = "/home/felipe/work/MyPackages/DAHAD/examples/WISEJ0259/mastDownload/JWST/jw02046-o003_t003_miri_f560w/jw02046-o003_t003_miri_f560w_i2d.fits"
    #path_data = "/home/felipe/work/StrongLensingModeling/SFstarred/notebooks/HST_IMAGES_0214_2105/WG0214-2105_F160W_drz_sci.fits"
    #path_data = "/home/felipe/work/Data/mast_data/WFIJ2033/JWST/mastDownload/JWST/jw01198-o004_t004_nircam_clear-f115w/jw01198-o004_t004_nircam_clear-f115w_i2d.fits"
    #remove_star: 5 (light contamination from other source), 8 (it is one of the QSO images); 9 (large residuals) 
    data_F160W = DataStarred(path_data)
    return (data_F160W,)


@app.cell
def _(data_F160W):
    data_F160W.cut_out_lens(plot=True, refine_centers=True, centroid_box_size=7, centroid_method="2dg",num_pix=100)
    return


@app.cell
def _(data_F160W):
    stat_stars = data_F160W.detect_stars(star_num_pix=40,detec_fwhm=None,threshold=None,verbose=True,make_plots=True,
                                         n_keep=20,
                                         binary_dilation_iteration=20,use_gaia=True,
                                         gaia_gmag_limit=20,gaia_radius=None,)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])

    mo.mpl.interactive(fig)   # makes it interactive with zoom/pan
    return


if __name__ == "__main__":
    app.run()
