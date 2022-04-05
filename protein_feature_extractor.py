import gzip
import os
import shutil
import subprocess
import sys
import time
import os.path as osp

import pandas as pd
import wget
from aaindex.aaindex import aaindex

AA_INDEX_AAs = "ALRKNMDFCPQSETGWHYIV"
MISSING_VAL = "NA"


def psiblast_arg_builder(sequence):
    # Note, this is based on
    # https://github.com/DeepRank/PSSMGen/blob/master/pssmgen/pssm.py

    args = {
        'query': "query.fasta",
        'out_ascii_pssm': "response.ascii_pssm",
        'out_pssm': "response.pssm",
        'out': "response.homologs",
        'gapextend': 1,
        'db': osp.join('..', '..', 'blastdb', 'uniprot_sprot'),  # Local SwissProt database
        'num_threads': 2,
        'evalue': 1e-4,
        'comp_based_stats': 'T',
        'max_target_seqs': 2000,
        'num_iterations': 3,
        'outfmt': 7,
    }

    if len(sequence) < 30:
        args['word_size'] = 2
    else:
        args['word_size'] = 3

    if len(sequence) < 35:
        args['gapopen'] = 9
    elif len(sequence) < 85:
        args['gapopen'] = 10
    else:
        args['gapopen'] = 11

    if len(sequence) < 35:
        args['matrix'] = 'PAM30'
    elif len(sequence) < 50:
        args['matrix'] = 'PAM70'
    elif len(sequence) < 85:
        args['matrix'] = 'BLOSUM80'
    else:
        args['matrix'] = 'BLOSUM62'

    return args


def make_pssm(sequence):
    timestamp = time.time()
    tmp_dir = osp.join("tmp", f"psiblast_{timestamp}")
    os.makedirs(tmp_dir)
    with open(osp.join(tmp_dir, "query.fasta"), "w") as f:
        f.write(f">query\n{sequence}\n")

    args = psiblast_arg_builder(sequence)

    if osp.exists(args['out_pssm']):
        print("PSI-BLAST PSSM already exists, skipping")
        return

    if not osp.exists('blastdb'):
        print("Downloading BLAST database...")
        os.makedirs('blastdb')
        # TODO Compare TrEMBL and SwissProt
        SWISS_PROT = "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
        TrEMBL = "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz"
        wget.download(SWISS_PROT, osp.join("blastdb", "uniprot_sprot.fasta.gz"))
        print("Unzipping...")
        with gzip.open(osp.join("blastdb", "uniprot_sprot.fasta.gz"), "rb") as f_in:
            with open(osp.join("blastdb", "uniprot_sprot.fasta"), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(osp.join("blastdb", "uniprot_sprot.fasta.gz"))
        print("Making the BLAST database...")
        subprocess.call("makeblastdb -in uniprot_sprot.fasta -dbtype prot -out uniprot_sprot", shell=True, cwd="blastdb")
        os.remove(osp.join("blastdb", "uniprot_sprot.fasta"))

    executable = "psiblast"

    cmd = f"{executable} -save_pssm_after_last_round " + " ".join(f"-{k} {v}" for k, v in args.items())

    print("Running PSI-BLAST...")
    print(cmd)
    subprocess.call(cmd, shell=True, cwd=tmp_dir)

    return tmp_dir


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 protein_feature_extractor.py <output_name.csv> <protein_sequence>")
        sys.exit(1)

    output_file = sys.argv[2].strip()
    sequence = sys.argv[2].upper().strip()

    if len([a for a in sequence if a not in AA_INDEX_AAs]) > 0:
        print("WARNING: Non-standard amino acid(s) found in sequence, this can effect feature output.")

    # Psi blast to make PSSM
    pssm_dir = make_pssm(sequence)
    pssm_file = osp.join(pssm_dir, "response.ascii_pssm")
    with open(pssm_file, "r") as f:
        for l in f:
            l = l.strip()
            ...
        # Note, broadcast kappa and lambda to every row since it is at the end for the protein as a whole

    # Delete pssm dir after parsing output
    shutil.rmtree(pssm_dir)

    # TODO: Get secondary structure predictions

    # Selected feature subsets
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
