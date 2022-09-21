"""
Implementation of the optimized Ising-Model described in [1]
written in JAX for GPU by Albert Alonso, Niels Bohr Institute.

[1] "A Performance Study of the 2D Ising Model on GPUs"
    https://arxiv.org/abs/1906.06297

Copyright (C) 2022 Albert Alonso

Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.
"""
import jax
import jax.numpy as jnp
import jax.random as jr


def apply_stencil(a, b, kernel, index):
    nn = (a @ kernel) + (kernel @ b)
    nn = nn.at[:,:,index,:].add(jnp.roll(b[:,:,-1-index,:], shift=2*index+1, axis=0))
    nn = nn.at[:,:,:,index].add(jnp.roll(b[:,:,:,-1-index], shift=2*index+1, axis=1))
    return nn

def ising_mcmc(key, shape, beta, niters):
    kernel = jnp.eye(shape[-1], k=-1, dtype=int) + jnp.eye(shape[-1], k=1, dtype=int)
    sections = [jnp.ones(shape)] * 4

    def metropolis_step(key, section, A, B, idx):
        probabilities = jr.uniform(key, shape)
        nn = apply_stencil(A, B, kernel, idx)
        acceptance_ratio = jnp.exp(-2 * beta * nn * section)
        flips = probabilities < acceptance_ratio
        return section - (2 * flips * section)

    def update_lattice(lattice, key):
        a1, b1, b2, a2 = lattice

        # Find the new spin values for both regions with the same "spin color".
        key1, key2 = jr.split(key, 2)
        r1 = metropolis_step(key1, a1, b1, b2, idx=0)
        r2 = metropolis_step(key2, a2, b1, b2, idx=-1)
        return [b1, r1, r2, b2], None

    keys = jr.split(key, niters//2)
    sections, _ = jax.lax.scan(update_lattice, init=sections, xs=keys)
    return jnp.array(sections)

