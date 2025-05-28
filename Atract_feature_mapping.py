# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 10:29:47 2025

@author: Guilherme Kundlatsch
"""

import itertools
import math
import pandas as pd

def shannon_entropy(counts):
    """
    Compute the Shannon entropy (in bits) given the counts of each nucleotide.
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
    Count how many times consecutive nucleotides differ in the sequence.
    """
    changes = 0
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            changes += 1
    return changes

def main():
    nucleotides = ['A', 'C', 'G', 'U']
    data = []
    
    # Generate every possible 8-nucleotide sequence (4^8 = 65,536 possibilities)
    for seq_tuple in itertools.product(nucleotides, repeat=8):
        seq = ''.join(seq_tuple)
        
        # Compute raw features.
        # A_Tract_state-change: count transitions over the full 8-nt sequence.
        sc = state_change_count(seq)
        # Normalize state-changes by dividing by 7.
        sc_norm = round(sc / 7, 4)
        
        # A%_6_A_tract: fraction of A's in the first 6 nucleotides.
        a_pct = round(seq[:6].count('A') / 6, 4)
        
        # C%_6_A_tract: fraction of C's in the first 6 nucleotides.
        c_pct = round(seq[:6].count('C') / 6, 4)
        
        # Entropia_A_tract: Shannon entropy for the full sequence.
        counts = (seq.count('A'), seq.count('C'), seq.count('G'), seq.count('U'))
        entropy = shannon_entropy(counts)
        # Normalize entropy by dividing by 2 (maximum is log2(4)=2 bits)
        entropy_norm = round(entropy / 2, 4)
        
        data.append({
            'A_Tract_state-change': sc_norm,
            'A%_6_A_tract': a_pct,
            'C%_6_A_tract': c_pct,
            'Entropia_A_tract': entropy_norm
        })
    
    # Convert list of dictionaries to DataFrame.
    df = pd.DataFrame(data)
    # Optionally, remove duplicate rows to have only unique combinations of feature values.
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Export the DataFrame to an Excel file.
    df.to_csv("Atract_feature_mapping_normalized.csv", index=False)
    print("Exported normalized Atract features to Atract_feature_mapping_normalized.xlsx")

if __name__ == "__main__":
    main()
