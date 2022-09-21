"""
Script to compute the magnetization at different temperatures of a grid.

Copyright (C) 2022 Albert Alonso

Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.
"""
import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from ising import ising_mcmc

parser = argparse.ArgumentParser()
parser.add_argument("--size", '-x', type=int, default=40, help="number of lattice tiles")
parser.add_argument("--tile", '-y', type=int, default=128, help="number of elements in tile")
parser.add_argument("--niters", '-n', type=int, default=100, help="number of trial iterations")
parser.add_argument("--seed", '-s', type=int, default=1234, help="seed for random number generation")
parser.add_argument("--Tmin",  type=float, default=1, help="Mininum temperature")
parser.add_argument("-Tmax", type=float, default=3.5, help="Maxinum temperature")
parser.add_argument("-Tpoints", type=int, default=100, help="Number of points")
parser.add_argument("--quiet", '-q', action='store_true', help="Do not show the figure")

args = parser.parse_args()

key = jr.PRNGKey(args.seed)
keys = jr.split(key, args.Tpoints)

temperatures = jnp.linspace(args.Tmin, args.Tmax, args.Tpoints)
subgrid_shape = (args.size // 2, args.size // 2, args.tile, args.tile)

print(f'Computing the ising model for T=[{args.Tmin},{args.Tmax}) in {args.Tpoints} points')
print(f'For a grid of size ({args.size*args.tile}, {args.size*args.tile})')

@jax.vmap
def compute_magnetization(T, key):
    beta = 1 / T
    tiles = ising_mcmc(key, shape=subgrid_shape, beta=beta, niters=args.niters)
    return jnp.mean(tiles)

Ts = temperatures.reshape((10, -1))
Ks = keys.reshape((10, -1, 2))
tic = time.time()
magnetizations = jnp.array([compute_magnetization(t,k) for (t, k) in zip(Ts, Ks)])
magnetizations = jax.block_until_ready(magnetizations)
print(f'Elapsed time: {time.time() - tic:.2f}s')

if not args.quiet:
    fig = plt.figure()
    plt.plot(temperatures.ravel(), magnetizations.ravel())
    plt.show()
