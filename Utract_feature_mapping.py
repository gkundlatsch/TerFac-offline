# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 10:38:50 2025

@author: Guilherme Kundlatsch
"""

import itertools
import math
import pandas as pd

def shannon_entropy(counts):
    """
    Compute the Shannon entropy (in bits) given a tuple of nucleotide counts.
    """
    total = sum(counts)
    H = 0.0
    for count in counts:
        if count > 0:
            freq = count / total
            H -= freq * math.log2(freq)
    return H

def state_change_count(seq):
    """
    Count how many times adjacent nucleotides differ within the sequence.
    """
    changes = 0
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            changes += 1
    return changes

def main():
    nucleotides = ['A', 'C', 'G', 'U']
    features_set = set()
    total_sequences = 4 ** 12  # 16,777,216 possible sequences.
    count = 0

    print("Starting enumeration of 12-nucleotide U-tracts...")
    for seq_tuple in itertools.product(nucleotides, repeat=12):
        seq = ''.join(seq_tuple)

        # 1. U%_10_U_tract: fraction of U's in the first 10 nucleotides.
        u_pct_10 = round(seq[:10].count('U') / 10, 4)
        
        # 2. U%_6_U_tract: fraction of U's in the first 6 nucleotides.
        u_pct_6 = round(seq[:6].count('U') / 6, 4)
        
        # 3. A%_6_U_tract: fraction of A's in the first 6 nucleotides.
        a_pct_6 = round(seq[:6].count('A') / 6, 4)
        
        # 4. C%_U_tract: fraction of C's in the entire 12 nucleotides.
        c_pct = round(seq.count('C') / 12, 4)
        
        # 5. U_Tract_state-change: count of state changes normalized by 11.
        sc = state_change_count(seq)
        sc_norm = round(sc / 11, 4)
        
        # 6. Entropia_U_tract: Shannon entropy normalized by 2.
        counts = (seq.count('A'), seq.count('C'), seq.count('G'), seq.count('U'))
        entropy = shannon_entropy(counts)
        entropy_norm = round(entropy / 2, 4)
        
        # Create the feature tuple.
        feature_tuple = (u_pct_10, u_pct_6, a_pct_6, c_pct, sc_norm, entropy_norm)
        features_set.add(feature_tuple)
        
        count += 1
        if count % 1000000 == 0:
            print(f"Processed {count} sequences out of {total_sequences}...")
    
    # Convert the set to a sorted list and then to a DataFrame.
    features_list = sorted(features_set)
    df = pd.DataFrame(features_list, columns=[
        "U%_10_U_tract",
        "U%_6_U_tract",
        "A%_6_U_tract",
        "C%_U_tract",
        "U_Tract_state-change",
        "Entropia_U_tract"
    ])
    
    # Export the DataFrame to an Excel file.
    df.to_csv("Utract_feature_mapping_normalized.csv", index=False)
    print("Export completed! Unique normalized U-tract features exported to Utract_feature_mapping_normalized.xlsx")

if __name__ == "__main__":
    main()
