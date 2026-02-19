### File containing necessary constants for excalibur
### Defining the units of choice for the program
### Available unit systems for now are SI and Cosmological units, which are defined arbitrarily to be Msun, Gyr and Gpc.
### About the natural units system: it is defined by setting c = G = 1, and keeping the same base units for mass, length and
### time as in the cosmological system.

c_si = 299792458                                             # Celerity of light in m/s
G_si = 6.67430e-11                                           # Newton's gravitational constant in m^3 kg^-1 s^-2
one_Msun_si = 1.98847e30                                     # Solar mass in kg
one_pc_si = 3.085677581e16                                   # Parsec in meters
one_kpc_si = 1e3 * one_pc_si                                 # Kiloparsec in meters
one_Mpc_si = 1e6 * one_pc_si                                 # Megaparsec in meters
one_Gpc_si = 1e9 * one_pc_si                                 # Gigaparsec in meters
one_yr_si = 365.25 * 24 * 3600                               # Year in seconds
one_Myr_si = 1e6 * one_yr_si                                 # Megayear in seconds
one_Gyr_si = 1e9 * one_yr_si                                 # Gigayear in seconds

###################################################################
################# Change the unit system here #####################
###################################################################

unit_system = "si"


if unit_system == "si" :
    c = c_si                                                 # Celerity of light in m/s
    G = G_si                                                 # Newton's gravitational constant in m^3 kg^-1 s^-2
    one_Msun = one_Msun_si                                   # Solar mass in kg
    one_pc = one_pc_si                                       # Parsec in meters
    one_kpc = one_kpc_si                                     # Kiloparsec in meters
    one_Mpc = one_Mpc_si                                     # Megaparsec in meters
    one_Gpc = one_Gpc_si                                     # Gigaparsec in meters
    one_yr = one_yr_si                                       # Year in seconds
    one_Myr = one_Myr_si                                     # Megayear in seconds
    one_Gyr = one_Gyr_si                                     # Gigayear in seconds
elif unit_system == "cosmo" :
    c = c_si * one_Gyr_si / one_Gpc_si                       # Celerity of light in cosmological units Gpc/Gyr
    G = G_si * one_Gyr_si**2 * one_Msun_si / one_Gpc_si**3   # Gravitational constant in Gpc^3 Msun^-1 Gyr^-2
    one_Msun = 1                                             # Msun is the base mass unit 
    one_pc = 1e-9                                            # Parsec in terms of the base length unit, the Gpc
    one_kpc = 1e3 * one_pc                                   # Kiloparsec in terms of Gpc
    one_Mpc = 1e6 * one_pc                                   # Megaparsec in terms of Gpc
    one_Gpc = 1                                              # Gigaparsec is the base length unit
    one_yr = 1e-9                                            # Year in terms of the base time unit, the Gyr
    one_kyr = 1e3 * one_yr                                   # Kiloyear in terms of Gyr
    one_Myr = 1e6 * one_yr                                   # Megayear in terms of Gyr
    one_Gyr = 1                                              # Gigayear is the base time unit      
elif unit_system == "natural" : 
    c = 1                                                    # Celerity of light in natural units
    G = 1                                                    # Gravitational constant in natural units
    one_Msun = 1                                             # Msun is the base mass unit 
    one_pc = 1e-9                                            # Parsec in terms of the base length unit, the Gpc
    one_kpc = 1e3 * one_pc                                   # Kiloparsec in terms of Gpc
    one_Mpc = 1e6 * one_pc                                   # Megaparsec in terms of Gpc
    one_Gpc = 1                                              # Gigaparsec is the base length unit
    one_yr = 1e-9                                            # Year in terms of the base time unit, the Gyr
    one_kyr = 1e3 * one_yr                                   # Kiloyear in terms of Gyr
    one_Mpc = 1e6 * one_yr                                   # Megayear in terms of Gyr
    one_Gyr = 1                                              # Gigayear is the base time unit
