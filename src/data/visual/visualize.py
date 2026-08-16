"""
Quick visualizer for py-pde FileStorage (.hdf5) output.

Usage:
    python visualize_output.py path/to/output.hdf5
    python visualize_output.py path/to/output.hdf5 --save plot.png
    python visualize_output.py path/to/output.hdf5 --frame -1   # plot only the last frame
    python visualize_output.py path/to/output.hdf5 --movie movie.mp4

For a 1D field, defaults to a kymograph (space vs. time heatmap).
For a 2D field, defaults to plotting the final frame (use --movie for an animation).
"""

import argparse
import sys

import matplotlib.pyplot as plt
import pde


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a py-pde FileStorage HDF5 file"
    )
    parser.add_argument("filename", help="Path to the .hdf5 file")
    parser.add_argument(
        "--save", help="Save the plot to this path instead of/as well as showing it"
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Plot a single frame by index (e.g. -1 for the last frame)",
    )
    parser.add_argument(
        "--movie",
        help="Render a movie to this path instead of a static plot (e.g. movie.mp4)",
    )
    args = parser.parse_args()

    store = pde.FileStorage(args.filename)

    if len(store) == 0:
        print(f"'{args.filename}' contains no stored frames.")
        sys.exit(1)

    grid = store.grid
    print(f"Loaded '{args.filename}': {len(store)} frames, grid={grid}")

    # Movie mode (works for 1D or 2D)
    if args.movie:
        pde.movie(store, args.movie)
        print(f"Movie saved to {args.movie}")
        return

    # Single-frame mode
    if args.frame is not None:
        times = list(store.times)
        items = list(store.items())
        t, field = items[args.frame]
        ax = field.plot(title=f"t = {t:.4f}")
        if args.save:
            plt.savefig(args.save, dpi=200, bbox_inches="tight")
            print(f"Saved to {args.save}")
        else:
            plt.show()
        return

    # Default: kymograph for 1D fields, final-frame plot for anything else
    if grid.dim == 1:
        ax = pde.plot_kymograph(store, filename=args.save)
        if not args.save:
            plt.show()
        print("Displayed kymograph (space vs. time).")
    else:
        t, field = list(store.items())[-1]
        field.plot(title=f"Final frame, t = {t:.4f}")
        if args.save:
            plt.savefig(args.save, dpi=200, bbox_inches="tight")
            print(f"Saved to {args.save}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
