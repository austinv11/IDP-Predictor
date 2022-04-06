import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
from shutil import which

import wget


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
        cmd = "psi-cd-hit.pl "

        args = {
            # Input file
            '-i ': "sequences.fasta",
            # Output name
            '-o ': "clusters",
            # Clustering threshold
            '-c ': 0.25,
            # Threads per blast job
            '-blp': 2
        }

        # Not available on windows, run through WSL
        if sys.platform == "win32":
            cmd = "wsl.exe " + cmd

        cmd += " ".join([f"{k} {v}" for k, v in args.items()])

        print(f"Running PSI-CD-HIT:")
        print(cmd)

        subprocess.call(cmd, shell=True, cwd=temp_dir)

        # Parse clusters

        #shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
