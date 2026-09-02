import argparse, os, sys, time
import h5py
import healpy as hp
import numpy as np
from mpi4py import MPI

# IMPORTANT: The absolute paths in this file are examples from the author's
# machine. Replace them with paths on your own computer or cluster before use.
# AKRA_DIR must point to the repository root and currently needs a trailing "/"
# because the import paths below are assembled by string concatenation.

AKRA_DIR = "/home/yshi/work/work_gravity/akra_series_local/"
sys.path.extend([AKRA_DIR + "utils/", AKRA_DIR + "core/"])
from gaussianfield import get_cl
from akra_full import sph_gamma2kappa, sph_kappa2gamma
from akra_full_cg import KappaRec_sphere_fast

COMM, RANK, SIZE = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
# Machine-specific DES Y3 data location. Update both the directory and filename
# for your local copy of the survey maps.
SKYMAP_DIR = "/home/yshi/Data/desy3_data2/"
DATA_FILE = "desy3_shear_maps_sharp_nside2048_zmin0.0_zmax1.5.hdf5"
BIN_INDEX, NSIDE_DATA, NEFF_THRESHOLD, SIGMA_E = 0, 2048, 2.0, 0.26


def parse_args():
    parser = argparse.ArgumentParser(description="Generate noisy shear realizations and reconstruct them with KS and AKRA.")
    parser.add_argument("--n-realizations", type=int, default=10)
    parser.add_argument("--start-realization", type=int, default=0)
    parser.add_argument("--stop-realization", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=1044)
    parser.add_argument("--nside", type=int, default=1024)
    parser.add_argument("--lmax", type=int, default=None)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--cgtol", type=float, default=2e-3)
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_survey_geometry(nside_out):
    with h5py.File(os.path.join(SKYMAP_DIR, DATA_FILE), "r") as f:
        group = f[list(f.keys())[BIN_INDEX]]
        w_sum, w2_sum = group["w_sum"][:], group["w2_sum"][:]
    neff_native = np.divide(w_sum**2, w2_sum, out=np.zeros_like(w_sum, dtype=float), where=w2_sum > 0)
    mask_native = neff_native > NEFF_THRESHOLD * hp.nside2pixarea(NSIDE_DATA, degrees=True) * 3600.0
    w_sum[~mask_native], w2_sum[~mask_native] = 0.0, 0.0
    w_sum_out, w2_sum_out = hp.ud_grade(w_sum, nside_out, power=-2), hp.ud_grade(w2_sum, nside_out, power=-2)
    mask = hp.ud_grade(mask_native.astype(float), nside_out) > 0.5
    neff = np.divide(w_sum_out**2, w2_sum_out, out=np.zeros_like(w_sum_out), where=w2_sum_out > 0)
    neff[~mask] = 0.0
    return mask, neff


def make_inputs(cl_kappa, mask, neff, nside, lmax, signal_seed, gamma1_noise_seed, gamma2_noise_seed):
    np.random.seed(signal_seed)
    kappa_true = hp.alm2map(hp.synalm(cl_kappa, lmax=lmax, new=True), nside=nside)
    gamma1_signal, gamma2_signal = sph_kappa2gamma(kappa_true, nside, lmax=lmax)
    sigma_n = np.zeros_like(neff)
    valid = mask & (neff > 0)
    sigma_n[valid] = SIGMA_E / np.sqrt(neff[valid])
    gamma1_noise = np.random.default_rng(gamma1_noise_seed).normal(size=mask.size) * sigma_n
    gamma2_noise = np.random.default_rng(gamma2_noise_seed).normal(size=mask.size) * sigma_n
    gamma1_signal, gamma2_signal = gamma1_signal * mask, gamma2_signal * mask
    gamma1 = np.stack([gamma1_signal, gamma1_noise, gamma1_signal + gamma1_noise])
    gamma2 = np.stack([gamma2_signal, gamma2_noise, gamma2_signal + gamma2_noise])
    return kappa_true, gamma1, gamma2


def reconstruct(gamma1, gamma2, mask, neff, nside, lmax, lam,  maxiter):
    kappa_ks, kappa_akra = np.empty_like(gamma1), np.empty_like(gamma1)
    for data_index in range(3):
        kappa_ks[data_index] = sph_gamma2kappa(gamma1[data_index], gamma2[data_index], nside, lmax=lmax)
        reconstruction = KappaRec_sphere_fast(gamma1[data_index], gamma2[data_index], mask, lmax=lmax, verbose=False, neff=neff)
        kappa_akra[data_index] = reconstruction.sphere_AKRA(lam=lam, maxiter=maxiter)
    return kappa_ks, kappa_akra


def save_realization(output_file, gamma1, gamma2, kappa_true, kappa_ks, kappa_akra, mask, attrs):
    temporary_file = output_file + ".tmp.rank%d" % RANK
    with h5py.File(temporary_file, "w") as f:
        for name, values in (("gamma1", gamma1), ("gamma2", gamma2), ("kappa_ks", kappa_ks), ("kappa_akra", kappa_akra), ("kappa_true", kappa_true), ("mask", mask)): f.create_dataset(name, data=values)
        for name, value in attrs.items(): f.attrs[name] = value
        f.attrs["index_0"], f.attrs["index_1"], f.attrs["index_2"] = "without noise", "noise", "noise + signal"
    os.replace(temporary_file, output_file)


def main():
    args = parse_args()
    if args.n_realizations < 1: raise ValueError("--n-realizations must be positive")
    stop_realization = args.n_realizations if args.stop_realization is None else args.stop_realization
    if not 0 <= args.start_realization < stop_realization <= args.n_realizations: raise ValueError("Require 0 <= start-realization < stop-realization <= n-realizations")
    # Without --output-dir, results are written under the machine-specific
    # SKYMAP_DIR above. Prefer an explicit path when running elsewhere.
    lmax = 2 * args.nside if args.lmax is None else args.lmax
    output_dir = args.output_dir or os.path.join(SKYMAP_DIR, "desy3_sim_withNoise_N%d_seed%d" % (args.n_realizations, args.base_seed))
    if RANK == 0:
        os.makedirs(output_dir, exist_ok=True)
        mask, neff = load_survey_geometry(args.nside)
        ell = np.arange(lmax + 1)
        cl_kappa = get_cl(ell=ell)
        cl_kappa[:2] = 0.0
        print("MPI ranks=%d, total realizations=%d, range=[%d,%d), nside=%d, lmax=%d" % (SIZE, args.n_realizations, args.start_realization, stop_realization, args.nside, lmax), flush=True)
        print("Output directory: %s" % output_dir, flush=True)
    else:
        mask, neff, cl_kappa = None, None, None
    mask, neff, cl_kappa = COMM.bcast(mask, root=0), COMM.bcast(neff, root=0), COMM.bcast(cl_kappa, root=0)
    COMM.Barrier()
    for realization in range(args.start_realization + RANK, stop_realization, SIZE):
        output_file = os.path.join(output_dir, "realization%03d.hdf5" % realization)
        if os.path.exists(output_file) and not args.overwrite:
            print("Rank %d skipping existing realization %d" % (RANK, realization), flush=True)
            continue
        start = time.time()
        signal_seed, gamma1_noise_seed, gamma2_noise_seed = args.base_seed + realization, args.base_seed + 1000000 + realization, args.base_seed + 2000000 + realization
        kappa_true, gamma1, gamma2 = make_inputs(cl_kappa, mask, neff, args.nside, lmax, signal_seed, gamma1_noise_seed, gamma2_noise_seed)
        kappa_ks, kappa_akra = reconstruct(gamma1, gamma2, mask, neff, args.nside, lmax, args.lam, args.maxiter)
        attrs = {"realization": realization, "group_start": args.start_realization, "group_stop": stop_realization, "nside": args.nside, "lmax": lmax, "base_seed": args.base_seed, "signal_seed": signal_seed, "gamma1_noise_seed": gamma1_noise_seed, "gamma2_noise_seed": gamma2_noise_seed, "sigma_e": SIGMA_E, "neff_threshold_arcmin2": NEFF_THRESHOLD, "akra_lam": args.lam,  "akra_maxiter": args.maxiter}
        save_realization(output_file, gamma1, gamma2, kappa_true, kappa_ks, kappa_akra, mask, attrs)
        print("Rank %d finished realization %d in %.1f s: %s" % (RANK, realization, time.time() - start, output_file), flush=True)
    COMM.Barrier()
    if RANK == 0: print("Finished realization range [%d,%d)." % (args.start_realization, stop_realization), flush=True)


if __name__ == "__main__": main()
