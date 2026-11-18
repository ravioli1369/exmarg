"""The tables of a store written once to a cache folder (exmarg.search.store.write_cache), so that the jobs of a machine
map them (analyse.py / plant.py --mmap with the cache folder as the store) and share one copy in the page cache instead
of each holding a private page-locked one.  The tables are written with the unbias correction (store.unbias), as
analyse.py and plant.py read them by default; --as-built writes them as built, for comparisons (their --as-built);
--unbiased names the default and changes nothing.
The cache records its kind (cache.json: unbiased), and the search refuses a cache of the other kind.

    python scripts/real_data/store_cache.py tables/redshift /scratch/cache/redshift [--readers 4] [--as-built]

About 0.75 GB of disk per node (40 GB for the 53 nodes of the published store); without --as-built about 4 GB of memory
per reader.  Nothing is done when the folder already holds the complete cache of the same files and kind.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from exmarg.search.store import TABLE_KINDS, write_cache


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('store', help='folder of the redshift tables (bd_*.npz)')
    p.add_argument('cache', help='folder of the cache (created)')
    p.add_argument('--readers', type=int, default=4, help='tables written at once')
    p.add_argument('--as-built', dest='as_built', action='store_true',
                   help='write the tables as built, without the unbias correction (comparisons; analyse.py, plant.py '
                        '--as-built)')
    p.add_argument('--unbiased', action='store_true',
                   help='the tables with the unbias correction, the default: changes nothing (not with --as-built)')
    a = p.parse_args(argv)
    if a.unbiased and a.as_built:
        p.error('--as-built writes the tables as built: not with --unbiased')
    t = time.time()
    write_cache(a.store, a.cache, a.readers, unbiased=not a.as_built)
    print('%s: cache of %s, tables %s, in %.0f s' % (a.cache, a.store, TABLE_KINDS[not a.as_built], time.time() - t),
          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
