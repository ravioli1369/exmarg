"""The tables of a store written once to a cache folder (exmarg.search.store.write_cache), so that the jobs of a machine
map them (analyse.py / plant.py --mmap with the cache folder as the store) and share one copy in the page cache instead
of each holding a private page-locked one.  The tables are written as built, as the search reads them by default;
--unbiased writes them with the unbias correction (store.unbias), for comparisons (analyse.py / plant.py --unbias).
The cache records its kind (cache.json: unbiased), and the search refuses a cache of the other kind.

    python scripts/real_data/store_cache.py tables/redshift /scratch/cache/redshift [--readers 4] [--unbiased]

About 0.75 GB of disk per node (40 GB for the 53 nodes of the published store); with --unbiased about 4 GB of memory per
reader.  Nothing is done when the folder already holds the complete cache of the same files and kind.
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
    p.add_argument('--unbiased', action='store_true',
                   help='write the tables with the unbias correction (comparisons; analyse.py, plant.py --unbias)')
    a = p.parse_args(argv)
    t = time.time()
    write_cache(a.store, a.cache, a.readers, unbiased=a.unbiased)
    print('%s: cache of %s, tables %s, in %.0f s' % (a.cache, a.store, TABLE_KINDS[a.unbiased], time.time() - t),
          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
