import sys
import os.path as osp

import pandas as pd
from aaindex.aaindex import aaindex

AA_INDEX_AAs = "ALRKNMDFCPQSETGWHYIV"
MISSING_VAL = "NA"


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 protein_feature_extractor.py <output_name.csv> <protein_sequence>")
        sys.exit(1)

    output_file = sys.argv[2].strip()
    sequence = sys.argv[2].upper().strip()

    if len([a for a in sequence if a not in AA_INDEX_AAs]) > 0:
        print("WARNING: Non-standard amino acid(s) found in sequence, this can effect feature output.")

    aa_feature_df = pd.read_csv("all_selected_features.csv", header=0)
    feat2vals = dict()
    features = list(aa_feature_df.features.values)
    for feat in features:
        feat2vals[feat] = []
        values = aaindex[feat]['values']
        for aa in sequence:
            feat2vals[feat].append(values.get(aa, MISSING_VAL))

    final_df = pd.DataFrame(list(sequence) + [feat2vals[feat] for feat in features], index=["sequence"] + features)
    final_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
