#### usual imports
import numpy as np

#### excalibur imports 
from excalibur.core.constants import *
from excalibur.metrics.schwarzschild_metric import SchwarzschildMetric
from excalibur.metrics.schwarzschild_metric_cartesian import SchwarzschildMetricCartesian
from excalibur.metrics.schwarzschild_metric import SchwarzschildMetric
from excalibur.integration.integrator import Integrator
from excalibur.photon.photons import Photons, Photon
from excalibur.io.filename_utils import generate_trajectory_filename
from excalibur.core.coordinates import spherical_to_cartesian, cartesian_to_spherical, cartesian_velocity_to_spherical

from excalibur.objects.spherical_mass import spherical_mass
from excalibur.grid.grid import Grid
from excalibur.grid.interpolator import Interpolator
from excalibur.metrics.perturbed_flrw_metric import PerturbedFLRWMetric
from excalibur.core.cosmology import LCDM_Cosmology

#### defining the schwarzschild metric
M = 10**15 * one_Msun
R = 3 * one_Mpc
center = np.array([0.5,0.5,0.5]) * one_Gpc

metric = SchwarzschildMetric(mass=M, radius=R, center=center, coords="spherical")

#### defining integration parameters
observer_position_cart = np.array([0.01,0.01,0.01]) * one_Gpc
# SchwarzschildMetric expects positions in spherical coordinates (t, r, theta, phi).
# Keep a clean cartesian copy for vector directions.
observer_position_sph = cartesian_to_spherical(*(observer_position_cart - center))

# Central direction for the cone must live in the same 3D space as the direction generator
# (which is cartesian). Use the cartesian vector from observer to mass center.
observation_vector_cart = center - observer_position_cart
observer_mass_distance = np.linalg.norm(observation_vector_cart)

num_steps = 5000
travel_distance = 2*observer_mass_distance
dx_norm = travel_distance/num_steps
dlambda = (dx_norm/c) 

mode = "sequential"
integration_scheme = "rk4"

integrator = Integrator(metric=metric, dt=dlambda, mode=mode, integrator=integration_scheme)

#### defining initial photon states
n_photons = 200
cone_angle = np.deg2rad(45)  # 5 degrees

photons = Photons(metric=metric)
# Build the 4D origin in spherical coordinates for the metric.
origin_sph_4d = np.array([0.0, *observer_position_sph], dtype=float)
photons.generate_cone_grid(
    n_photons=n_photons,
    origin=origin_sph_4d,
    central_direction=observation_vector_cart,
    cone_angle=cone_angle,
    direction_basis="cartesian",
)

positions = photons.positions()
directions = photons.directions()


print("Schwarzschild - photon[5] null rel error:", photons.photons[5].photon_norm(metric))
print("Schwarzschild - photon[5] |sqrt(u.u)|:", photons.photons[5].photon_norm_abs(metric))

norms = np.array([])
for photon in photons.photons:
    norms = np.append(norms, photon.photon_norm(metric))

print("mean photon norm:", np.mean(norms))
print("max photon norm:", np.max(norms))
print("min photon norm:", np.min(norms))
print("std photon norm:", np.std(norms))
print("how much photons have 0 norm?", np.sum(norms==0))
print("how many photons have near zero norm?", np.sum(np.isclose(norms,0, atol=1e-14)))


#### just for some comparisons with flrw

cosmo = LCDM_Cosmology(H0=70, Omega_m=0.3, Omega_r=0, Omega_lambda=0.7)

N = 500
grid_size = one_Gpc
shape = (N,N,N)
dx = dy = dz = grid_size / N
spacing = (dx, dy, dz)
grid = Grid(shape=shape, spacing=spacing)
interp = Interpolator(grid)

x = y = z = np.linspace(0, grid_size, N, endpoint=False)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

mass_object = spherical_mass(mass=M, radius=R, center=center)
phi = mass_object.potential(X,Y,Z)

grid.add_field('Phi',phi)

flrw_metric = PerturbedFLRWMetric(cosmo, grid, interp)

flrwphotons = Photons(metric=flrw_metric)

flrwphotons.generate_cone_grid(
    n_photons=n_photons,
    origin=np.array([0.0, *observer_position_cart], dtype=float),
    central_direction=observation_vector_cart,
    cone_angle=cone_angle,
    direction_basis="cartesian",
)

print("FLRW ######################")
flrwnorms = np.array([])
for photon in flrwphotons.photons:
    flrwnorms = np.append(flrwnorms, photon.photon_norm(flrw_metric))
print("FLRW mean photon norm:", np.mean(flrwnorms))
print("FLRW max photon norm:", np.max(flrwnorms))
print("FLRW min photon norm:", np.min(flrwnorms))
print("FLRW std photon norm:", np.std(flrwnorms))
print("FLRW how many photons ?", len(flrwnorms))
print("FLRW how much photons have 0 norm?", np.sum(flrwnorms==0))
print("FLRW how many photons have near zero norm?", np.sum(np.isclose(flrwnorms,0, atol=1e-14)))
print(flrwnorms)
#### 


