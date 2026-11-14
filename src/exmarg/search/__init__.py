"""
the analysis of segments with the stored tables, on the CPU or the GPU.

    store = load_redshift_store(store_dir)                  # once: the tables of every signal power node
    bank = mass_bank(bank_3col_file, segments, device)      # once per frequency grid and noise curve
    res = pe_intrinsic(segments, bank, store, prior)        # M, q, chi, u, ln_lambda, ln_b (distance integrated)

    bank = mass_bank(bank_file, segments, device)           # equal mass, zero spin
    res = pe_redshift(segments, bank, store)                # m1, z, ln_lambda, ln_b

device is 'cpu' (numpy and numba, every core) or 'gpu' (cupy).  For the
GPU, load the store with pinned=True for fast copies to the device.

    res = pe_redshift(segments, bank, store, single=True)   # also ln_lambda_H, ln_lambda_L, ln_glitch_H, ...

single.py holds the single-detector evidences (docs/glitch_likelihood.md):
the columns, the histogram of a trajectory and the glitch time weights.
"""

from .segments import D0, read_segments
from .bank import mass_bank
from .store import load_redshift_store
from .pe import pe_redshift, pe_intrinsic, distance_weights
from .single import (RHO_EDGES, kernel, ln_kernel, response_distribution, glitch_columns, merger_columns,
                     histogram, ln_evidence, ln_merger_at, glitch_time_series, glitch_time_weights)
