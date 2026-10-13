"""Scripts written for CosTaL 0.0.2 run unchanged on CosTaL+ through costal_plus.legacy.

    python examples/migrate_from_costal.py
"""

import sys

from costal_plus import legacy as CosTaL  # was: import CosTaL
from cytometry import synthetic


def main(argv):
    raw, truth = synthetic()
    labels = CosTaL.find_communities(raw, pp_method="mass", nbr_num=10, resolution=0.8, seed=0)
    scores = CosTaL.return_scores(truth, labels)
    print(f"{max(labels) + 1} clusters; AMI {scores[0]:.3f}, ARI {scores[1]:.3f}, FlowCAP F1 {scores[2]:.3f}")
    return labels


if __name__ == "__main__":
    main(sys.argv)
