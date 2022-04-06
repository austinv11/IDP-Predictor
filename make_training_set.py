import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import time

import numpy as np
import wget

from protein_feature_extractor import extract_features


def download_json():
    if not osp.exists("disprot.json"):
        print("Downloading complete DisProt database...")
        wget.download("https://disprot.org/api/search?release=2022_03&show_ambiguous=true&show_obsolete=false&format=json", "disprot.json")


def main():
    TRAIN_SPLIT = 0.7

    download_json()

    with open("disprot.json", 'rb') as f:
        disprot = json.load(f)['data']

    os.makedirs("dataset", exist_ok=True)

    if not osp.exists(osp.join("dataset", "clusters.csv")):
        print("Generating Clusters from sequences...")
        print("Make sure PSI-CD-HIT is installed and in your PATH!")

        # Dont use osp.join since we are using wsl potentially
        temp_dir = f"tmp/cdhit_{time.time()}"
        os.makedirs(temp_dir)
        with open(temp_dir + "/sequences.fasta", "w") as f:
            for disprot_entry in disprot:
                name = disprot_entry['acc']
                sequence = disprot_entry['sequence']
                f.write(f">{name}\n{sequence}\n")

        # We are gonna use psi-cd-hit to cluster the sequences
        # So that we an use a lower threshold for the clustering
        # Following recommendations from here to do iterative clustering:
        # https://github.com/weizhongli/cdhit/wiki/3.-User's-Guide#protein-clustering
        cdhitcmd = "cd-hit "
        cdhit_args = {
            # Input file
            '-i': "sequences.fasta",
            # Output database
            '-o': "preclust90",
            # Identity
            '-c': 0.9,
            # Word Size
            '-n': 5,
            # Accurate mode
            '-g': 1,
            # Local coverage mode
            '-G': 0,
            # Alignment coverage required for shorter sequence
            '-aS': 0.8,
            # Disable fasta output from cluster file
            '-d': 0,
            # Print alignment overlap
            '-p': 1,
            # Threads
            '-T': 2,
            # Unlimited memory
            '-M': 0
        }
        cdhitcmd90 = cdhitcmd + " ".join([f"{k} {v}" for k, v in cdhit_args.items()])

        cdhit_args['-i'] = "preclust90"
        cdhit_args['-o'] = "preclust60"
        cdhit_args['-c'] = 0.6
        cdhit_args['-n'] = 4
        cdhitcmd60 = cdhitcmd + " ".join([f"{k} {v}" for k, v in cdhit_args.items()])

        psicmd = "psi-cd-hit.pl "
        psiargs = {
            # Input database
            '-i': "preclust60",
            # Output name
            '-o': "preclust25",
            # Clustering threshold
            '-c': 0.25,
            # Threads per blast job
            '-blp': 2
        }
        psicmd += " ".join([f"{k} {v}" for k, v in psiargs.items()])

        # hierachical clustering
        clust_rev_cmd = "clstr_rev.pl {}.clstr {}.clstr > {}.clstr"

        # Not available on windows, run through WSL
        if sys.platform == "win32":
            psicmd = "wsl.exe " + psicmd
            cdhitcmd90 = "wsl.exe " + cdhitcmd90
            cdhitcmd60 = "wsl.exe " + cdhitcmd60
            clust_rev_cmd = "wsl.exe " + clust_rev_cmd

        clust_rev_cmd1 = clust_rev_cmd.format("preclust90", "preclust60", "preclust90-60")
        clust_rev_cmd2 = clust_rev_cmd.format("preclust90-60", "preclust25", "clusters")

        print(f"Pre-Clustering at 90% identity...")
        print(cdhitcmd90)
        subprocess.call(cdhitcmd90, shell=True, cwd=temp_dir)

        print(f"Pre-Clustering at 60% identity...")
        print(cdhitcmd60)
        subprocess.call(cdhitcmd60, shell=True, cwd=temp_dir)

        print(f"Running PSI-CD-HIT:")
        print(psicmd)
        subprocess.call(psicmd, shell=True, cwd=temp_dir)

        print("Combining Clusters...")

        print(clust_rev_cmd1)
        subprocess.call(clust_rev_cmd1, shell=True, cwd=temp_dir)
        print(clust_rev_cmd2)
        subprocess.call(clust_rev_cmd2, shell=True, cwd=temp_dir)

        print("Reading Cluster output!")

        # Parse clusters
        with open(osp.join(temp_dir, "clusters.clstr"), "r") as fin:
            with open(osp.join("dataset", "clusters.csv"), "w") as fout:
                fout.write("sequence,cluster")
                curr_clust = None
                for l in fin:
                    l = l.strip()
                    if len(l) == 0:
                        continue
                    if l.startswith(">"):
                        curr_clust = l.split()[1]
                    else:
                        # Example line: "0	3430aa, >P06935... *"
                        idname = l.split()[2].replace(">", "").replace("...", "")
                        fout.write(f"\n{idname},{curr_clust}")

        shutil.rmtree(temp_dir)
    else:
        print("Clusters already generated, skipping cluster generation...")

    if not osp.exists(osp.join("dataset", "train_split.csv")):
        # Generate test/train split
        print("Generating test/train split...")
        train_set = np.random.choice(len(disprot), int(len(disprot) * TRAIN_SPLIT), replace=False)

        seq2cluster = dict()
        train_seqs = []
        train_clusters = set()
        test_seqs = []
        with open(osp.join("dataset", "clusters.csv"), "r") as f:
            first = True
            index = 0
            for l in f:
                if first:
                    first = False
                    continue
                l = l.strip()
                split = l.split(',')
                cluster = int(split[1])
                if index in train_set:
                    train_seqs.append(split[0])
                    train_clusters.add(cluster)
                else:
                    test_seqs.append(split[0])
                seq2cluster[split[0]] = cluster
                index += 1

        removed = []
        for i in range(len(test_seqs)):
            test_cluster = seq2cluster[test_seqs[i]]
            if test_cluster in train_clusters:
                removed.append(test_seqs[i])
        print("Tossing out {} of the {} sequences due to cluster overlap with the training set".format(len(removed), len(test_seqs)))

        with open(osp.join("dataset", "train_split.csv"), "w") as f:
            f.write("sequence,test")
            for seq in train_seqs:
                f.write(f"\n{seq},0")
            for seq in test_seqs:
                if seq in removed:
                    continue
                f.write(f"\n{seq},1")

    else:
        print("Test/train split already generated, skipping split generation...")

    print("Generating features for the sequences... This can take a while!")
    train_dir = osp.join("dataset", "train")
    test_dir = osp.join("dataset", "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Generate features for the sequences
    seq2split = dict()  # Map sequence to split, 1 = test, 0 = train
    with open(osp.join("dataset", "train_split.csv"), "r") as f:
        first = True
        for l in f:
            if first:
                first = False
                continue
            l = l.strip()
            split = l.split(",")
            seq2split[split[0]] = int(split[1])

    total_seqs = len(seq2split)
    curr_seqs = 1
    for prot in disprot:
        acc = prot['acc']
        # Make sure its not dropped
        if acc not in seq2split:
            continue

        print("## Generating features for {} ({}/{})".format(acc, curr_seqs, total_seqs))
        curr_seqs += 1

        output_dir = train_dir if seq2split[acc] == 0 else test_dir
        output_file = osp.join(output_dir, acc)

        # Generate features
        if not osp.exists(output_file + ".parquet"):
            print(f"Generating features for {acc}...")

            seq = prot['sequence'].strip()
            idp_regions = set()
            for region in prot['disprot_consensus']['full']:
                idp_regions.add(f"{region['start']}-{region['end']}")

            print("Getting features for {}...".format(acc))

            extract_features(output_file, seq, idp_regions)


if __name__ == "__main__":
    main()
