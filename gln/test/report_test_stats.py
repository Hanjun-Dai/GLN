from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import os
import sys

if __name__ == '__main__':
    files = os.listdir(sys.argv[1])

    best_val = 0.0
    best_test = None
    for fname in files:
        if 'val-' in fname and 'summary' in fname:
            f_test = os.path.join(sys.argv[1], 'test' + fname[3:])
            if not os.path.isfile(f_test):
                continue
            with open(os.path.join(sys.argv[1], fname), 'r') as f:
                f.readline()
                top1 = float(f.readline().strip().split()[-1].strip())
                if top1 > best_val:
                    best_val = top1
                    best_test = f_test
    assert best_test is not None
    with open(best_test, 'r') as f:
        for row in f:
            print(row.strip())
    print(best_test)
