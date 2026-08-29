"""
Generate synthetic PDE training data and save it in HDF5 format.

This script solves a specified PDE (see below for supported equations)
over a (currently preset) domain, optionally adds Gaussian noise,
and writes the results to an .hdf5 file. It also saves the
command-line arguments used to a separate text file for reproducibility.

May add support for generating data from the saved text files later,
would be useful for data too large for Github.

Command-line usage:
    python data_gen.py <destination> <noise_amount> <pde_name> [pde_args...]

Arguments:
    destination     Base name for output files (without extension).
    noise_amount    Standard deviation of Gaussian noise to add (0 for no noise).
    pde_name        One of: black_scholes, allen_cahn, diff1d, diff2d.
    pde_args        Dependent on specified PDE (see below).

PDE-specific arguments:
    black_scholes   option_name strike_price sigma r t_max
                    (option_name: EUROCALL or EUROPUT)
    allen_cahn      epsilon t_max
    diff1d          diffusion_coefficient t_max
    diff2d          diffusion_coefficient t_max

Outputs:
    <destination>.hdf5         : HDF5 storage of the solved field snapshots.
    <destination>_args.txt     : Text file recording the arguments used.
"""

import sys
from pathlib import Path

import pde

from gen.funcs import apply_noise, pdes

REPO_ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        sigma = float(sys.argv[2])
        donoise = sigma != 0
        pde_func = pdes[sys.argv[3]]
        output = pde_func(*sys.argv[4:])
        if donoise:
            output = apply_noise(output, sigma)
    except (IndexError, TypeError, KeyError) as e:
        print("""
        Usage: 
        python data_gen.py [destination] [noise amount] [PDE Name] [PDE Arguments] [t_max]
        for no noise, set the parameter to 0.
        Valid PDEs:                         Arguments 
        black_scholes                       option name, strike price, sigma, r
        (currently supports 
        EUROCALL and EUROPUT)
        allen_cahn                          epsilon
        diff1d          <- 1D diffusion     diffusion coefficient
        diff2d          <- 2d diffusion     diffusion coefficient
        """)
        print(e)
        import traceback

        print("".join(traceback.format_tb(e.__traceback__)))
        sys.exit()

    out_path = REPO_ROOT / "figs" / f"{filename}.hdf5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_store = pde.FileStorage(str(out_path), info=output.info)
    output.apply(lambda field: field, out=file_store)

    args_path = REPO_ROOT / "figs" / f"{filename}_args.txt"
    with open(args_path, "w") as f:
        f.write(", ".join(sys.argv[2:]))
