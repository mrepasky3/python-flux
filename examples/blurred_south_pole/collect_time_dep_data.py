#!/usr/bin/env python
import json_numpy as json
import scipy.sparse
import sys
import glob
import pyvista as pv
import os

import meshio
import numpy as np
from tqdm import tqdm

from spice_util import get_sunvec
from flux.compressed_form_factors_nmf import CompressedFormFactorMatrix
from flux.model import ThermalModel
from flux.shape import CgalTrimeshShapeModel, get_surface_normals

import argparse
import arrow

parser = argparse.ArgumentParser()
parser.add_argument('--compression_type', type=str, default="svd",choices=["nmf","snmf","wsnmf",
    "svd","ssvd",
    "rand_svd","rand_ssvd","rand_snmf",
    "aca", "brp", "rand_id",
    "saca","sbrp","rand_sid",
    "stoch_radiosity",
    "true_model"])
parser.add_argument('--max_area', type=float, default=3.0)
parser.add_argument('--outer_radius', type=int, default=80)
parser.add_argument('--sigma', type=float, default=5.0)
parser.add_argument('--tol', type=float, default=1e-1)

parser.add_argument('--min_depth', type=int, default=1)
parser.add_argument('--max_depth', type=int, default=0)

parser.add_argument('--nmf_max_iters', type=int, default=int(1e4))
parser.add_argument('--nmf_tol', type=float, default=1e-2)

parser.add_argument('--k0', type=int, default=40)

parser.add_argument('--p', type=int, default=5)
parser.add_argument('--q', type=int, default=1)

parser.add_argument('--nmf_beta_loss', type=int, default=2, choices=[1,2])

parser.add_argument('--compress_sparse', action='store_true')

parser.set_defaults(feature=False)

args = parser.parse_args()

# some useful routines
# transform cartesian to spherical (meters, radians)
def cart2sph(xyz):

    rtmp = np.linalg.norm(np.array(xyz).reshape(-1, 3), axis=1)
    lattmp = np.arcsin(np.array(xyz).reshape(-1, 3)[:, 2] / rtmp)
    lontmp = np.arctan2(np.array(xyz).reshape(-1, 3)[:, 1], np.array(xyz).reshape(-1, 3)[:, 0])

    return rtmp, lattmp, lontmp

def sind(x):
    return np.sin(np.deg2rad(x))


def cosd(x):
    return np.cos(np.deg2rad(x))

def project_stereographic(lon, lat, lon0, lat0, R=1):
    """
    project cylindrical coordinates to stereographic xy from central lon0/lat0
    :param lon: array of input longitudes (deg)
    :param lat: array of input latitudes (deg)
    :param lon0: center longitude for the projection (deg)
    :param lat0: center latitude for the projection (deg)
    :param R: planetary radius (km)
    :return: stereographic projection xy coord from center (km)
    """

    cosd_lat = cosd(lat)
    cosd_lon_lon0 = cosd(lon - lon0)
    sind_lat = sind(lat)

    k = (2. * R) / (1. + sind(lat0) * sind_lat + cosd(lat0) * cosd_lat * cosd_lon_lon0)
    x = k * cosd_lat * sind(lon - lon0)
    y = k * (cosd(lat0) * sind_lat - sind(lat0) * cosd_lat * cosd_lon_lon0)

    return x, y

# ============================================================
# main code

compression_type = args.compression_type
max_area_str = str(args.max_area)
outer_radius_str = str(args.outer_radius)
sigma_str = str(args.sigma)
tol_str = "{:.0e}".format(args.tol)

max_depth = args.max_depth if args.max_depth != 0 else None


if compression_type == "true_model":
    FF_dir = "true_{}_{}_{}".format(max_area_str, outer_radius_str, sigma_str)

elif compression_type == "stoch_radiosity":
    FF_dir = "stoch_rad_{}_{}_{}_{}k0".format(max_area_str, outer_radius_str, sigma_str,
        args.k0)

elif compression_type == "svd":
    FF_dir = "{}_{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, sigma_str, args.tol,
        args.k0)

elif compression_type == "ssvd":
    FF_dir = "{}_{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, sigma_str, args.tol,
        args.k0)

elif compression_type == "rand_svd":
    FF_dir = "{}_{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, sigma_str, args.tol,
        args.p, args.q, args.k0)

elif compression_type == "rand_ssvd":
    FF_dir = "{}_{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, sigma_str, args.tol,
        args.p, args.q, args.k0)

elif compression_type == "nmf":
    FF_dir = "{}_{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "klnmf", max_area_str, outer_radius_str, sigma_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)

elif compression_type == "snmf":
    FF_dir = "{}_{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "sklnmf", max_area_str, outer_radius_str, sigma_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)

elif compression_type == "rand_snmf":
    FF_dir = "{}_{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, sigma_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.p, args.q, args.k0)

elif compression_type == "wsnmf":
    FF_dir = "{}_{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "wsklnmf", max_area_str, outer_radius_str, sigma_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)

elif compression_type == "aca":
    FF_dir = "{}_{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, sigma_str, args.tol,
        args.k0)

elif compression_type == "brp":
    FF_dir = "{}_{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, sigma_str, args.tol,
        args.k0)

elif compression_type == "rand_id":
    FF_dir = "{}_{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, sigma_str, args.tol,
        args.p, args.q, args.k0)

elif compression_type == "saca":
    FF_dir = "{}_{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, sigma_str, args.tol,
        args.k0)

elif compression_type == "sbrp":
    FF_dir = "{}_{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, sigma_str, args.tol,
        args.k0)

elif compression_type == "rand_sid":
    FF_dir = "{}_{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, sigma_str, args.tol,
        args.p, args.q, args.k0)


if not (compression_type == "true_model" or compression_type == "stoch_radiosity" or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k") and args.min_depth != 1:
    FF_dir += "_{}mindepth".format(args.min_depth)

if not (compression_type == "true_model" or compression_type == "stoch_radiosity" or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k") and max_depth is not None:
    FF_dir += "_{}maxdepth".format(max_depth)

if not (compression_type == "true_model" or compression_type == "stoch_radiosity" or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k") and args.compress_sparse:
    savedir += "_cs"


FF_dir = "results/"+FF_dir
if not os.path.exists(FF_dir):
    print("PATH DOES NOT EXIST "+FF_dir)
    assert False
savedir = FF_dir + "/T_frames"
if not os.path.exists(savedir):
    os.mkdir(savedir)


# read shapemodel and form-factor matrix generated by make_compressed_form_factor_matrix.py
if compression_type == 'true_model' or compression_type == 'stoch_radiosity':
    path = FF_dir+f'/FF_{max_area_str}_{outer_radius_str}.npz'
    FF = scipy.sparse.load_npz(path)
    V = np.load(f'blurred_pole_verts_{max_area_str}_{outer_radius_str}_{sigma_str}.npy')
    F = np.load(f'blurred_pole_faces_{max_area_str}_{outer_radius_str}_{sigma_str}.npy')
    N = get_surface_normals(V, F)
    N[N[:, 2] > 0] *= -1
    shape_model = CgalTrimeshShapeModel(V, F, N)
else:
    path = FF_dir+f'/FF_{max_area_str}_{outer_radius_str}_{tol_str}_{compression_type}.bin'
    FF = CompressedFormFactorMatrix.from_file(path)
    shape_model = FF.shape_model

print('  * loaded form factor matrix and (cartesian) shape model')

utc0 = '2011 MAR 01 00:00:00.00'
utc1 = '2011 MAR 31 00:00:00.00'
num_frames = 3000
stepet = 2592000/3000
sun_vecs = get_sunvec(utc0=utc0, utc1=utc1, stepet=stepet, path_to_furnsh="simple.furnsh",
                      target='SUN', observer='MOON', frame='MOON_ME')
t = np.linspace(0, 2592000, num_frames + 1)

D = sun_vecs/np.linalg.norm(sun_vecs, axis=1)[:, np.newaxis]
D = D.copy(order='C')

print('  * got sun positions from SPICE')

z = np.linspace(0, 3e-3, 31)

print('  * set up thermal model')
thermal_model = ThermalModel(
    FF, t, D,
    F0=np.repeat(1365, len(D)), rho=0.11, method='1mvp',
    z=z, T0=100, ti=120, rhoc=9.6e5, emiss=0.95,
    Fgeotherm=0.2, bcond='Q', shape_model=shape_model)

Tmin, Tmax = np.inf, -np.inf
vmin, vmax = 90, 310

sim_start_time = arrow.now()
for frame_index, T in tqdm(enumerate(thermal_model), total=D.shape[0], desc='thermal models time-steps'):
    path = savedir+"/T{:03d}.npy".format(frame_index)
    np.save(path, T)
sim_duration = (arrow.now()-sim_start_time).total_seconds()
print('  * thermal model run completed in {:.2f} seconds'.format(sim_duration))

np.save(savedir+f'/sim_duration.npy', np.array(sim_duration))