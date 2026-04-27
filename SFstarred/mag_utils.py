import numpy as np 


header_values ={'F814W': {'PHOTFLAM': 1.516962e-19,
  'PHOTZPT': -21.1,
  'PHOTPLAM': 8032.93505,
  'PHOTFNU': 3.2921571625e-07,
  'PHOTBW': 665.299255},
 'F475X': {'PHOTFLAM': 1.561348275e-19,
  'PHOTZPT': -21.1,
  'PHOTPLAM': 4940.7244,
  'PHOTFNU': 1.2787481875e-07,
  'PHOTBW': 660.4336249999999},
 'F160W': {'PHOTFLAM': 1.9429001e-20,
  'PHOTZPT': -21.1,
  'PHOTPLAM': 15369.176,
  'PHOTFNU': 1.5308434e-07,
  'PHOTBW': 826.25085}}


###########To get mag#########################
def get_stmag(flux, photflam, photzpt):
   flux = np.asarray(flux)
   scalar_input = False
   if flux.ndim == 0:
      flux = flux[None] 
      scalar_input = True
   flux = flux * photflam
   mag = -2.5 * np.log10(flux) + photzpt

   if scalar_input:
       return np.squeeze(mag)

   return mag


def get_abmag(flux, photflam, photzpt, photplam):
   stmag = get_stmag(flux, photflam, photzpt)

   return stmag - 5. * np.log10(photplam) + 2.5 * np.log10(299792458e10) - 27.5