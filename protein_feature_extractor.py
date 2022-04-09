import gzip
import os
import shutil
import subprocess
import sys
import tarfile
import time
import os.path as osp
from statistics import mean
from zipfile import ZipFile

import pandas as pd
import wget
from aaindex.aaindex import aaindex

AA_INDEX_AAs = "ALRKNMDFCPQSETGWHYIV"
MISSING_VAL = "X"
# https://en.wikipedia.org/wiki/FASTA_format#Sequence_representation
AMBIGUOUS_TO_POSSIBLE = {
    "B": ['D', 'N'],
    "J": ['I', "L"],
    "Z": ['E', 'Q'],
    "O": ['K'],
    "U": ['C'],
    "X": list(AA_INDEX_AAs),
}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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
    os.makedirs(tmp_dir, exist_ok=True)
    with open(osp.join(tmp_dir, "query.fasta"), "w") as f:
        f.write(f">query\n{sequence}\n")

    args = psiblast_arg_builder(sequence)

    if osp.exists(args['out_pssm']):
        print("PSI-BLAST PSSM already exists, skipping")
        return

    if not osp.exists('blastdb'):
        print("Need to prepare BLAST database, this will take a long time!")
        print("Downloading BLAST database...")
        os.makedirs('blastdb', exist_ok=True)
        # TrEMBL is way slower than SwissProt, so we'll use SwissProt
        # Smaller swiss prot database
        SWISS_PROT = "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
        # Bigger trembl database
        TrEMBL = "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz"
        # Mid-sized database, much bigger than SWISS PROT but smaller than TREMBL
        UNIREF = "ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
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


def make_secondary_structs(sequence):
    # Using s4pred as a lightweight secondary structure prediction tool
    # In the future we may want to use psipred instead (difficult to install)

    if not osp.exists("s4pred-main"):
        print("Downloading S4PRED...")
        os.makedirs("tmp", exist_ok=True)
        wget.download("https://github.com/psipred/s4pred/archive/refs/heads/main.zip", osp.join("tmp", "s4pred.zip"))
        with ZipFile(osp.join("tmp", "s4pred.zip"), "r") as zip_ref:
            zip_ref.extractall()
        os.remove(osp.join("tmp", "s4pred.zip"))
        print("Downloading S4PRED weights...")
        wget.download("http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/weights.tar.gz", osp.join("s4pred-main", "weights.tar.gz"))
        with tarfile.open(osp.join("s4pred-main", "weights.tar.gz"), "r:gz") as tar:
            tar.extractall("s4pred-main")
        os.remove(osp.join("s4pred-main", "weights.tar.gz"))

    timestamp = time.time()
    tmp_file = osp.join("tmp", f"s4pred_{timestamp}.fasta")
    with open(tmp_file, "w") as f:
        f.write(f">query\n{sequence}\n")

    proc = subprocess.Popen(['python', osp.join('s4pred-main', 'run_model.py'), tmp_file], stdout=subprocess.PIPE)
    output = proc.communicate()[0]

    os.remove(tmp_file)

    return output.decode("utf-8")  # Psi-pred output


def extract_features(output_file, sequence, idp_ranges):
    # Expand IDP ranges to flat list of positions
    # Note, these ranges are 1-indexed, inclusive
    idp_positions = []
    linker_positions = []
    protein_binding_positions = []
    nucleic_binding_positions = []
    generic_binding_positions = []
    for idp_range in idp_ranges:
        split_tag = idp_range.split(":")
        split_range = split_tag[-1].split("-")
        tag = split_tag[0].lower()
        positions = list(range(int(split_range[0]), int(split_range[1])+1))
        if 'linker' in tag:
            linker_positions += positions
        elif 'protein' in tag:
            protein_binding_positions += positions
        elif 'nucleic' in tag:
            nucleic_binding_positions += positions
        elif 'binding' in tag:
            generic_binding_positions += positions
        idp_positions += positions

    positions = [i for i in range(1, len(sequence)+1)]
    disordered_labels = [int(i in idp_positions) for i in positions]
    generic_binding_labels = [int(i in generic_binding_positions) for i in positions]
    linker_labels = [int(i in linker_positions) for i in positions]
    protein_binding_labels = [int(i in protein_binding_positions) for i in positions]
    nucleic_binding_labels = [int(i in nucleic_binding_positions) for i in positions]

    if len([a for a in sequence if a not in AA_INDEX_AAs]) > 0:
        eprint("WARNING: Non-standard amino acid(s) found in sequence, this can effect feature output.")

    # Psi-blast to make PSSM
    pssm_dir = make_pssm(sequence)
    pssm_file = osp.join(pssm_dir, "response.ascii_pssm")
    pssm_aas = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    pssm_columns = [aa + '_Score' for aa in pssm_aas] + [aa + '_WeightedScore' for aa in pssm_aas] +\
                   ['InformationPerPos', 'RelativeWeightedMatches', 'K_StdUngapped', 'Lambda_StdUngapped',
                    'K_StdGapped', 'Lambda_StdGapped', 'K_PsiUngapped', 'Lambda_PsiUngapped',
                    'K_PsiGapped', 'Lambda_PsiGapped']

    std_ungapped = None
    std_gapped = None
    psi_ungapped = None
    psi_gapped = None
    pssm_data = []
    if osp.exists(pssm_file):
        with open(pssm_file, "r") as f:
            parsing_matrix = True
            last_pos = 1
            for l in f:
                l = l.strip()
                if parsing_matrix:
                    if len(l) == 0 and len(pssm_data) > 0:
                        parsing_matrix = False
                    else:
                        split_line = l.split()
                        try:
                            curr_pos = int(split_line[0])  # Need to see if psiblast skips positions
                            assert curr_pos > 0
                        except:
                            continue  # Skip this line since it didnt start with a position
                        if curr_pos != last_pos+1:
                            # Zero-fill the missing positions
                            for i in range(last_pos, curr_pos):
                                pssm_data.append([0.0] * (len(pssm_columns)-8))
                        pssm_data.append([float(a) for a in l.split()[2:]])  # Skip the position and letter

                        last_pos = curr_pos
                else:
                    if not (l.startswith('Standard') or l.startswith('PSI')):
                        continue
                    # Note, broadcast kappa and lambda to every row since it is at the end for the protein as a whole
                    split = l.split()
                    K = float(split[2])  # Lines start with 2 words
                    Lambda = float(split[3])
                    if l.startswith('Standard Ungapped'):
                        std_ungapped = (K, Lambda)
                    elif l.startswith('Standard Gapped'):
                        std_gapped = (K, Lambda)
                    elif l.startswith('PSI Ungapped'):
                        psi_ungapped = (K, Lambda)
                    elif l.startswith('PSI Gapped'):
                        psi_gapped = (K, Lambda)
                    else:
                        raise AssertionError("Unknown line: " + l)

        # Insert the kappa and lambda values
        for i in range(len(pssm_data)):
            pssm_data[i] = pssm_data[i] + [std_ungapped[0], std_ungapped[1], std_gapped[0], std_gapped[1],
                                           psi_ungapped[0], psi_ungapped[1], psi_gapped[0], psi_gapped[1]]
    else:
        # PSSM FAILED
        eprint("WARNING: PSSM failed, using default values")
        pssm_data = [[0.0] * len(pssm_columns)] * len(sequence)

    # Delete pssm dir after parsing output
    shutil.rmtree(pssm_dir)

    # S4Pred for secondary structure prediction
    psipred_matrix = make_secondary_structs(sequence)
    # Now, parse the psipred format output
    psipred_data = []
    psipred_columns = ['coil_prob', 'helix_prob', 'sheet_prob']
    last_pos = 0
    for l in psipred_matrix.splitlines():
        l = l.strip()
        if len(l) == 0 or l.startswith('#'):
            continue
        split = l.split()
        curr_pos = int(split[0])
        if curr_pos != last_pos+1:
            # Zero-fill the missing positions
            for i in range(last_pos, curr_pos):
                psipred_data.append([0.0] * len(psipred_columns))
        last_pos = curr_pos
        psipred_data.append([float(a) for a in split[3:]])

    # Selected feature subsets for biochemical properties
    # AAIndex1
    aa_feature_df = pd.read_csv("all_selected_features.csv", header=0)
    feat2vals = dict()
    features = list(aa_feature_df.features.values)
    for feat in features:
        feat2vals[feat] = []
        values = aaindex[feat]['values']
        for aa in sequence:
            if aa in AMBIGUOUS_TO_POSSIBLE:
                eprint("WARNING: Ambiguous amino acid " + aa + " in sequence, averaging properties of: " + "".join(AMBIGUOUS_TO_POSSIBLE[aa]))
                feat2vals[feat].append(mean([float(values.get(ambig, 0)) for ambig in AMBIGUOUS_TO_POSSIBLE[aa]]))
            else:
                feat2vals[feat].append(float(values.get(aa, 0)))

    feature_dict = {
        # Class Labels
        'is_disordered': disordered_labels,
        'is_binding': generic_binding_labels,
        'is_linker': linker_labels,
        'is_protein_binding': protein_binding_labels,
        'is_nucleic_acid_binding': nucleic_binding_labels,

        # Sequence Features
        'position': positions,
        'sequence': list(sequence),
        'sequence_length': [len(sequence)] * len(sequence),
    }

    for feat in features:
        feature_dict[feat] = feat2vals[feat]

    for i, pssm_column in enumerate(pssm_columns):
        feature_dict[pssm_column] = [row[i] for row in pssm_data]

    for i, psipred_column in enumerate(psipred_columns):
        feature_dict[psipred_column] = [row[i] for row in psipred_data]

    final_df = pd.DataFrame(feature_dict)
    #final_df.to_csv(output_file + ".csv", index=False)
    final_df.to_parquet(output_file + ".parquet", index=False, compression="brotli")


if __name__ == "__main__":
    if len(sys.argv) < 3 or 'help' in sys.argv[1]:
        print("Usage: python3 protein_feature_extractor.py <output_location> <protein_sequence> <IDP_range1> <IDP_range2> ...")
        print("Example IDP ranges: 1-10 12-20 22-30")
        print("Example Linker, Protein Binding, Nucleic Acid Binding, and Generic Binding ranges: linker:10-15 protein:60-70 nucleic:30-40 binding:55-60")
        sys.exit(1)

    output_file = sys.argv[1].strip()
    sequence = sys.argv[2].upper().strip()
    idp_ranges = sys.argv[3:]
