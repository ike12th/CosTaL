"""Scripts written for CosTaL 1.0 run unchanged on CosTaL 2.0.

    python examples/migrate_from_costal.py
"""

import sys

import CosTaL  # 1.0's import name, kept as an alias of `costal`
import costal
from cytometry import synthetic


def main(argv):
    raw, truth = synthetic()
    # a CosTaL 1.0 call, unchanged: preprocessing inside, a list of labels back
    labels = CosTaL.find_communities(raw, pp_method="mass", nbr_num=10, resolution=0.8, seed=0)
    scores = CosTaL.return_scores(truth, labels)
    print(f"1.0 call: {max(labels) + 1} clusters; AMI {scores[0]:.3f}, ARI {scores[1]:.3f}, FlowCAP F1 {scores[2]:.3f}")

    # the same with the 2.0 interface: preprocessing as its own step, a numpy array back
    new = costal.find_communities(costal.preprocess(raw, "mass"), k=10, resolution=0.8, seed=0)
    print(f"2.0 call:   identical labels: {new.tolist() == labels}")
    return labels


if __name__ == "__main__":
    main(sys.argv)
