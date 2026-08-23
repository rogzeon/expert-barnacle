import sys

import pde

from funcs import apply_noise, pdes

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

    file_store = pde.FileStorage(f"src/data/{filename}.hdf5", info=output.info)
    output.apply(lambda field: field, out=file_store)

    with open(f"src/data/{filename}_args.txt", "w") as f:
        f.write(", ".join(sys.argv[2:]))
