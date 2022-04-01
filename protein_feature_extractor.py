import sys
import os.path as osp

import pandas as pd

AA_INDEX_AAs = "ALRKNMDFCPQSETGWHYIV"


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 protein_feature_extractor.py <protein_sequence>")
        sys.exit(1)

    sequence = sys.argv[1].upper().strip()

    if len([a for a in sequence if a not in AA_INDEX_AAs]) > 0:
        print("WARNING: Non-standard amino acid(s) found in sequence, this can effect feature output.")

    aa_feature_df = pd.read_csv("all_selected_features.csv", header=0)


if __name__ == "__main__":
    main()
