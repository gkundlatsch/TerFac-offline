# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 10:48:53 2025

@author: Guilherme Kundlatsch
"""

import pandas as pd

def main():
    data = []
    min_length = 3
    max_length = 16

    # For each possible loop length, calculate the normalized Tamanho Loop.
    for L in range(min_length, max_length + 1):
        # Normalize the length: (L - 3) / (16 - 3) = (L - 3) / 13
        normalized_length = round((L - min_length) / (max_length - min_length), 4)
        
        # For each possible number of G's from 0 to L, calculate the %GC.
        # Note: %GC_Loop is defined as (# of G's) / L.
        for g_count in range(0, L + 1):
            gc_fraction = round(g_count / L, 4)
            data.append({
                "Tamanho Loop": normalized_length,
                "%GC_Loop": gc_fraction
            })
    
    # Remove duplicates (if any) and sort the DataFrame.
    df = pd.DataFrame(data)
    df = df.drop_duplicates().sort_values(["Tamanho Loop", "%GC_Loop"]).reset_index(drop=True)
    
    # Export the unique normalized feature combinations to an Excel file.
    df.to_csv("Loop_feature_mapping_normalized.csv", index=False)
    print("Exported normalized loop feature mapping to Loop_feature_mapping_normalized.xlsx")

if __name__ == "__main__":
    main()
