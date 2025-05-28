import pandas as pd
import math
import joblib
import numpy as np
from scipy.optimize import differential_evolution
import itertools
import warnings
warnings.filterwarnings("ignore")  # Nuclear option: suppresses ALL warnings


eval_count = 0
REPORT_INTERVAL = 1000

#########################
# Global Constants & Model Loading
#########################

desired_order = [
    "Tamanho Loop",
    "A%_6_A_tract",
    "C%_6_A_tract",
    "U%_10_U_tract",
    "U%_6_U_tract",
    "A%_6_U_tract",
    "C%_U_tract",
    "Tamanho Hairpin sem Loop",
    "%GC_Loop",
    "Entropia_A_tract",
    "Entropia_U_tract",
    "Entropia_HP_S_Loop",
    "A_Tract_state-change",
    "U_Tract_state-change",
    "HP_S_Loop_state_change",
    "GC_Inicial_Hairpin"
]

# Load the predictor model from disk.
model_data = joblib.load("terminator_strength_predictor.joblib")
model = model_data["model"]
print("Final feature order defined and model loaded successfully!")

max_hairpin_norm = None

#########################
# Mapping Functions
#########################

def load_atract_mapping(file_path="Atract_feature_mapping_normalized.csv"):
    df = pd.read_csv(file_path)
    feature_cols = ["A_Tract_state-change", "A%_6_A_tract", "C%_6_A_tract", "Entropia_A_tract"]
    for col in feature_cols:
        df[col] = df[col].astype(float).round(4)
    mapping = {}
    for _, row in df.iterrows():
        entropy = row["Entropia_A_tract"]
        combination = (row["A_Tract_state-change"], row["A%_6_A_tract"], row["C%_6_A_tract"])
        mapping.setdefault(entropy, []).append(combination)
    return mapping, df

def load_utract_mapping(file_path="Utract_feature_mapping_normalized.csv"):
    df = pd.read_csv(file_path)
    feature_cols = [
        "U%_10_U_tract",
        "U%_6_U_tract",
        "A%_6_U_tract",
        "C%_U_tract",
        "U_Tract_state-change",
        "Entropia_U_tract"
    ]
    for col in feature_cols:
        df[col] = df[col].astype(float).round(4)
    mapping = {}
    for _, row in df.iterrows():
        entropy = row["Entropia_U_tract"]
        combination = (
            row["U%_10_U_tract"],
            row["U%_6_U_tract"],
            row["A%_6_U_tract"],
            row["C%_U_tract"],
            row["U_Tract_state-change"]
        )
        mapping.setdefault(entropy, []).append(combination)
    return mapping, df

def load_loop_mapping(file_path="Loop_feature_mapping_normalized.csv"):
    df = pd.read_csv(file_path)
    feature_cols = ["Tamanho Loop", "%GC_Loop"]
    for col in feature_cols:
        df[col] = df[col].astype(float).round(4)
    mapping = {}
    for _, row in df.iterrows():
        loop_norm = row["Tamanho Loop"]
        gc_val = row["%GC_Loop"]
        mapping.setdefault(loop_norm, []).append(gc_val)
    return mapping, df

def load_hairpin_mapping(file_path="Hairpin_feature_mapping.csv"):
    df = pd.read_csv(file_path)
    feature_cols = [
        "Tamanho Hairpin sem Loop",
        "GC_Inicial_Hairpin",
        "Entropia_HP_S_Loop",
        "HP_S_Loop_state_change"
    ]
    for col in feature_cols:
        df[col] = df[col].astype(float).round(4)
    mapping = {}
    for _, row in df.iterrows():
        key = row["Tamanho Hairpin sem Loop"]
        combination = (
            row["GC_Inicial_Hairpin"],
            row["Entropia_HP_S_Loop"],
            row["HP_S_Loop_state_change"]
        )
        mapping.setdefault(key, []).append(combination)
    return mapping, df

# ----------------------
# Global mapping variables (loaded once at import time)
# ----------------------
atract_mapping, _  = load_atract_mapping()
utract_mapping, _  = load_utract_mapping()
loop_mapping, _    = load_loop_mapping()
hairpin_mapping, _ = load_hairpin_mapping()

# Precompute sorted keys
loop_keys = sorted(loop_mapping.keys())
a_keys    = sorted(atract_mapping.keys())
u_keys    = sorted(utract_mapping.keys())
h_keys    = sorted(hairpin_mapping.keys())


#########################
# Snapping Function
#########################

def snap_to_valid_candidate(candidate, loop_mapping, atract_mapping, utract_mapping, hairpin_mapping):
    """
    Given a candidate 16-element feature vector (ordered as in desired_order), snaps each group
    to the nearest valid discrete value using precomputed sorted keys.
    
    Candidate vector index mapping:
      0:  Tamanho Loop
      1:  A%_6_A_tract
      2:  C%_6_A_tract
      3:  U%_10_U_tract
      4:  U%_6_U_tract
      5:  A%_6_A_tract   (for U-tract)
      6:  C%_U_tract
      7:  Tamanho Hairpin sem Loop
      8:  %GC_Loop
      9:  Entropia_A_tract
      10: Entropia_U_tract
      11: Entropia_HP_S_Loop
      12: A_Tract_state-change
      13: U_Tract_state-change
      14: HP_S_Loop_state_change
      15: GC_Inicial_Hairpin
    """
    snapped = candidate.copy()
    global loop_keys, a_keys, u_keys, h_keys

    def euclidean_dist(t1, t2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(t1, t2)))

    # ----- Loop group -----
    # Use precomputed loop_keys.
    loop_key = min(loop_keys, key=lambda k: abs(k - candidate[0]))
    snapped[0] = loop_key
    allowed_gc = loop_mapping[loop_key]
    snapped[8] = min(allowed_gc, key=lambda v: abs(v - candidate[8]))

    # ----- A-tract group -----
    a_key = min(a_keys, key=lambda k: abs(k - candidate[9]))
    snapped[9] = a_key
    a_candidate_tuple = (candidate[12], candidate[1], candidate[2])
    valid_a_tuple = min(atract_mapping[a_key], key=lambda tup: euclidean_dist(tup, a_candidate_tuple))
    snapped[1]  = valid_a_tuple[1]
    snapped[2]  = valid_a_tuple[2]
    snapped[12] = valid_a_tuple[0]

    # ----- U-tract group -----
    u_key = min(u_keys, key=lambda k: abs(k - candidate[10]))
    snapped[10] = u_key
    u_candidate_tuple = (candidate[3], candidate[4], candidate[5], candidate[6], candidate[13])
    valid_u_tuple = min(utract_mapping[u_key], key=lambda tup: euclidean_dist(tup, u_candidate_tuple))
    snapped[3]  = valid_u_tuple[0]
    snapped[4]  = valid_u_tuple[1]
    snapped[5]  = valid_u_tuple[2]
    snapped[6]  = valid_u_tuple[3]
    snapped[13] = valid_u_tuple[4]

    # ----- Hairpin group -----
    h_key = min(h_keys, key=lambda k: abs(k - candidate[7]))
    snapped[7] = h_key
    hairpin_candidate_tuple = (candidate[15], candidate[11], candidate[14])
    valid_h_tuple = min(hairpin_mapping[h_key], key=lambda tup: euclidean_dist(tup, hairpin_candidate_tuple))
    snapped[11] = valid_h_tuple[1]
    snapped[14] = valid_h_tuple[2]
    snapped[15] = valid_h_tuple[0]

    return snapped

def cost_maximize(x):
    """
    Negative of the predicted strength so differential_evolution will maximize
    instead of minimize, with periodic progress logging.
    """
    global eval_count
    eval_count += 1

    # Log every REPORT_INTERVAL calls
    if eval_count % REPORT_INTERVAL == 0:
        print(f"[Maximize] Evaluations = {eval_count}")

    # snap to the nearest valid discrete feature vector
    x_valid = snap_to_valid_candidate(
        x, loop_mapping, atract_mapping, utract_mapping, hairpin_mapping
    )
    # enforce the hairpin‐size cap
    if x_valid[7] > max_hairpin_norm:
        return np.inf

    strength = model.predict(x_valid.reshape(1, -1))[0]
    # return negative for maximization
    return -strength


# --- Denormalization Helpers ---

def denormalize_feature(value, min_val, max_val):
    """Converts a normalized value [0,1] to its real-world value using a linear mapping."""
    return round(value * (max_val - min_val) + min_val, 4)

def denormalize_dataframe(df):
    """Denormalizes specific columns in the DataFrame based on predefined ranges."""
    ranges = {
        "Tamanho Loop": (3, 16),
        "Tamanho Hairpin sem Loop": (6, 49),
        "Entropia_A_tract": (0, 2),
        "Entropia_U_tract": (0, 2),
        "Entropia_HP_S_Loop": (0, 2),
        "A_Tract_state-change": (0, 7),
        "U_Tract_state-change": (0, 11),
        "HP_S_Loop_state_change": (2, 34),
        "GC_Inicial_Hairpin": (0, 12)
    }
    df_denorm = df.copy()
    for feature, (min_val, max_val) in ranges.items():
        if feature in df_denorm.columns:
            df_denorm[feature] = df_denorm[feature].apply(lambda x: denormalize_feature(x, min_val, max_val))
    return df_denorm

# --- Final Value Calculation Helpers ---

def calculate_final_values(df):
    """
    Multiplies normalized feature values by their specified factors to obtain final real-world values.
    """
    df_final = df.copy()
    if "A%_6_A_tract" in df_final.columns:
        df_final["A%_6_A_tract"] *= 6
    if "C%_6_A_tract" in df_final.columns:
        df_final["C%_6_A_tract"] *= 6
    if "U%_10_U_tract" in df_final.columns:
        df_final["U%_10_U_tract"] *= 10
    if "U%_6_U_tract" in df_final.columns:
        df_final["U%_6_U_tract"] *= 6
    if "A%_6_U_tract" in df_final.columns:
        df_final["A%_6_U_tract"] *= 6
    if "C%_U_tract" in df_final.columns:
        df_final["C%_U_tract"] *= 12
    if ("%GC_Loop" in df_final.columns) and ("Tamanho Loop" in df_final.columns):
        df_final["%GC_Loop"] = df_final["%GC_Loop"] * df_final["Tamanho Loop"]
    return df_final

def convert_columns_to_int(df, columns):
    """Rounds and converts the listed columns in the DataFrame to integers."""
    df_int = df.copy()
    for col in columns:
        if col in df_int.columns:
            df_int[col] = df_int[col].round().astype(int)
    return df_int

def shannon_entropy(counts):
    """
    Compute the Shannon entropy (in bits) given a distribution of counts.
    Returns the value rounded to 4 decimal places.
    counts: a tuple (count_A, count_C, count_G, count_U)
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    H = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            H -= p * math.log2(p)
    return round(H, 4)

def state_change_count(sequence):
    """
    Count how many times consecutive nucleotides differ in the sequence.
    """
    changes = 0
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i - 1]:
            changes += 1
    return changes

def generate_a_tract_by_target(entropy_target, a6_target, c6_target, state_change_target):
    """
    Finds an 8-mer A-tract such that:
      - Shannon entropy (full 8-mer) ≈ entropy_target (±0.01)
      - State-change count (full 8-mer) == state_change_target
      - In the last 6 bases, count('A') == a6_target and count('C') == c6_target
    """
    nucleotides = ['A','C','G','U']
    for seq_tuple in itertools.product(nucleotides, repeat=8):
        seq = ''.join(seq_tuple)
        # full-length features
        H = shannon_entropy(tuple(seq.count(nt) for nt in 'ACGU'))
        sc = state_change_count(seq)
        if not math.isclose(H, entropy_target, abs_tol=0.01): continue
        if sc != state_change_target:                         continue

        # last-6 counts
        tail = seq[-6:]
        if tail.count('A') != a6_target: continue
        if tail.count('C') != c6_target: continue

        return seq  # first one that matches
    return None

def gen_compositions(L):
    """
    Generate all 4-tuple compositions (n_A, n_C, n_G, n_U) that sum to L.
    """
    comps = []
    for n_A in range(L + 1):
        for n_C in range(L + 1 - n_A):
            for n_G in range(L + 1 - n_A - n_C):
                n_U = L - n_A - n_C - n_G
                comps.append((n_A, n_C, n_G, n_U))
    return comps

def compute_entropy(comp, L):
    """
    Compute Shannon entropy (in bits) for a composition.
    comp: a tuple (n_A, n_C, n_G, n_U) representing counts.
    L: the total length (half-length in our case).
    """
    H = 0.0
    for count in comp:
        if count > 0:
            p = count / L
            H -= p * math.log2(p)
    return H

def compute_state_change_range(comp, L):
    """
    For a given composition over L positions, compute:
      - raw_min: achieved when identical nucleotides are grouped.
        (raw_min = [number of nonzero counts] - 1)
      - raw_max: achievable when nucleotides are arranged to maximize changes.
        Here we use a heuristic:
             if max(comp) <= L - max(comp) + 1 then raw_max = L - 1,
             else raw_max = 2 * (L - max(comp)).
    """
    nonzero = sum(1 for x in comp if x > 0)
    raw_min = nonzero - 1
    m = max(comp)
    if m <= L - m + 1:
        raw_max = L - 1
    else:
        raw_max = 2 * (L - m)
    return raw_min, raw_max

def multiset_permutations(iterable):
    """
    Generate all unique orderings (permutations) of the elements in 'iterable',
    which may contain duplicates.
    
    This is a recursive generator that yields each unique permutation.
    """
    pool = tuple(iterable)
    n = len(pool)
    if n == 0:
        return
    counter = {}
    for item in pool:
        counter[item] = counter.get(item, 0) + 1

    def backtrack(path, counter):
        if len(path) == n:
            yield tuple(path)
            return
        for x in counter:
            if counter[x] > 0:
                counter[x] -= 1
                path.append(x)
                yield from backtrack(path, counter)
                path.pop()
                counter[x] += 1

    yield from backtrack([], counter)


def generate_u_tract_by_distribution(entropia_u, u_state_change, u10, u6, a6, c_total, entropy_tol=0.01, progress_freq=100000):
    """
    Searches for valid U-tract sequences (length = 12) by first optimizing the search space using 
    overall Shannon entropy and state-change filtering, then generating candidate orderings for each valid 
    composition and finally testing all the detailed conditions:
    
      - Overall Shannon entropy (from nucleotide counts) equals entropia_u (within tolerance).
      - Overall state-change count (over the full 12 nt) equals u_state_change.
      - The first 10 nucleotides contain exactly u10 U's.
      - The first 6 nucleotides contain exactly u6 U's and exactly a6 A's.
      - The total count of C's in the 12-nt sequence equals c_total.
    
    Returns:
      (found_sequence, all_trials)
      where found_sequence is the first candidate (string) that satisfies all conditions (or None)
      and all_trials is a list of dictionaries logging each trial.
    """
    tract_length = 12
    nucleotides = ['A', 'C', 'G', 'U']
    all_trials = []
    found_sequence = None
    trial_count = 0

    # --- Pre-filter compositions by overall entropy ---
    valid_entropy_comps = []
    for comp in gen_compositions(tract_length):
        H = compute_entropy(comp, tract_length)
        if abs(H - entropia_u) <= entropy_tol:
            valid_entropy_comps.append(comp)
    valid_entropy_comps = set(valid_entropy_comps)

    # --- Pre-filter compositions by overall state-change capability ---
    valid_state_comps = []
    for comp in gen_compositions(tract_length):
        raw_min, raw_max = compute_state_change_range(comp, tract_length)
        if raw_min <= u_state_change <= raw_max:
            valid_state_comps.append(comp)
    valid_state_comps = set(valid_state_comps)

    # Use the intersection of both filters.
    valid_comps = valid_entropy_comps.intersection(valid_state_comps)
    print(f"U-tract pre-filtering: {len(valid_comps)} valid compositions by entropy and state-change.")
    



    # --- Now, for each valid composition, generate candidate orderings ---
    for comp in valid_comps:
        ordering_list = []
        for nuc, count in zip(nucleotides, comp):
            ordering_list.extend([nuc] * count)

        # Generate all unique sequences (orderings) from this composition.
        for perm_tuple in multiset_permutations(ordering_list):
            trial_count += 1
            candidate_seq = ''.join(perm_tuple)
            
            trial = {}
            trial['sequence'] = candidate_seq
            # Compute overall nucleotide counts.
            count_A = candidate_seq.count('A')
            count_C = candidate_seq.count('C')
            count_G = candidate_seq.count('G')
            count_U = candidate_seq.count('U')
            trial['count_A'] = count_A
            trial['count_C'] = count_C
            trial['count_G'] = count_G
            trial['count_U'] = count_U
            distribution = (count_A, count_C, count_G, count_U)
            H_candidate = shannon_entropy(distribution)
            trial['entropy'] = H_candidate

            # Compute overall state-change.
            st_changes = state_change_count(candidate_seq)
            trial['state_changes'] = st_changes

            failure_reasons = []
            # Condition 1: Overall entropy must match.
            if not math.isclose(H_candidate, entropia_u, abs_tol=entropy_tol):
                failure_reasons.append(f"Entropy {H_candidate} != target {entropia_u}")
            # Condition 2: Overall state-change count.
            if st_changes != u_state_change:
                failure_reasons.append(f"State changes {st_changes} != target {u_state_change}")
            # Condition 3: First 10 nucleotides' U count.
            if candidate_seq[:10].count('U') != u10:
                failure_reasons.append(f"First 10 U count {candidate_seq[:10].count('U')} != {u10}")
            # Condition 4: First 6 nucleotides' U and A counts.
            if candidate_seq[:6].count('U') != u6:
                failure_reasons.append(f"First 6 U count {candidate_seq[:6].count('U')} != {u6}")
            if candidate_seq[:6].count('A') != a6:
                failure_reasons.append(f"First 6 A count {candidate_seq[:6].count('A')} != {a6}")
            # Condition 5: Overall C count.
            if candidate_seq.count('C') != c_total:
                failure_reasons.append(f"Total C count {candidate_seq.count('C')} != {c_total}")
            
            trial['failure_reasons'] = " | ".join(failure_reasons)
            trial['passed'] = (len(failure_reasons) == 0)
            all_trials.append(trial)
            
            if trial['passed']:
                print(f"Found valid U-tract after {trial_count} trials: {candidate_seq}")
                found_sequence = candidate_seq
                return found_sequence, all_trials
            
            if trial_count % progress_freq == 0:
                print(f"Tested {trial_count} candidates...")
    
    print("Search completed. No valid U-tract sequence found.")
    return None, all_trials


def generate_loop_sequence_by_features(df_final):
    """
    Generates a candidate loop sequence based on the final features.
    It extracts:
      - The denormalized 'Tamanho Loop' as the actual loop length (an integer between 3 and 16)
      - The target number of G's from '%GC_Loop' (which has been calculated as "fraction * Tamanho Loop")
    Then, it returns a candidate sequence of length L that contains exactly the target G count.
    The remaining positions will be filled with 'A'.
    
    Returns:
      A string representing the candidate loop sequence, or None if parameters are invalid.
    """
    try:
        # Extract the loop length from the final features (assumed to have been denormalized to an integer)
        loop_length = int(df_final["Tamanho Loop"].iloc[0])
    except Exception as e:
        print("Error extracting Tamanho Loop:", e)
        return None

    # Extract the absolute count of G's.
    # Note: In your final value calculations, you set:
    #   final_values["%GC_Loop"] = final_values["%GC_Loop"] * final_values["Tamanho Loop"]
    # So here, '%GC_Loop' is the number of G's.
    try:
        target_G = int(round(float(df_final["%GC_Loop"].iloc[0])))
    except Exception as e:
        print("Error extracting %GC_Loop:", e)
        return None

    # Basic checks:
    if target_G < 0 or target_G > loop_length:
        print("Invalid target G count:", target_G, "for loop length:", loop_length)
        return None

    # Generate a candidate loop sequence.
    # For simplicity, we'll generate the lexicographically first sequence that has exactly target_G G's.
    # The idea is: choose target_G positions (lowest indices) to be 'G' and fill the rest with 'A'.
    # More sophisticated methods could iterate over all possibilities.
    seq = ['A'] * loop_length
    # Choose the lowest target_G positions to be 'G', e.g. indices 0, 1, ..., target_G-1.
    for i in range(target_G):
        seq[i] = 'G'
    # Alternatively, if you prefer to have G's more concentrated at the end, you could pick indices from the end.
    # For example:
    # for i in range(loop_length - target_G, loop_length):
    #     seq[i] = 'G'
    return "".join(seq)


def valid_compositions_by_entropy(half_length, target_entropy, entropy_tol):
    valid = []
    comps = gen_compositions(half_length)
    for comp in comps:
        H = compute_entropy(comp, half_length)
        if abs(H - target_entropy) <= entropy_tol:
            valid.append(comp)
    
    # Add debugging to see how many valid compositions are found
    print(f"Valid compositions by entropy: {len(valid)}")
    return set(valid)

def valid_compositions_by_state_change(half_length, first_half_target_state_change):
    valid = []
    comps = gen_compositions(half_length)
    for comp in comps:
        raw_min, raw_max = compute_state_change_range(comp, half_length)
        if raw_min <= first_half_target_state_change <= raw_max:
            valid.append(comp)

    # Add debugging to see how many valid compositions are found
    print(f"Valid compositions by state change: {len(valid)}")
    return set(valid)


def initial_gc_count(sequence):
    """
    Count how many consecutive nucleotides at the start of the sequence are either G or C.
    """
    count = 0
    for nt in sequence:
        if nt in ['C', 'G']:
            count += 1
        else:
            break
    return count

def generate_second_half(first_half):
    """
    Generate the second half of the hairpin by reverse complementing the first half.
    Mapping: A <-> U and C <-> G.
    """
    complement = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
    return "".join(complement[nt] for nt in reversed(first_half))

# -----------------------
# Candidate Generator with Combined Filtering
# -----------------------

def generate_hairpin_first_half(half_length,
                                target_entropy,
                                full_target_state_changes,
                                target_gc_initial,
                                entropy_tol=0.01,
                                progress_freq=1000000):
    """
    Generates the first half of the hairpin stem (length = half_length) that meets:
      - overall Shannon entropy ≈ target_entropy (± entropy_tol)
      - capacity to achieve the full_target_state_changes
      - an initial run of G/C of length target_gc_initial
    """

    # Adjust the state-change target for the first half (assuming symmetry + junction)
    first_half_target_state_change = (full_target_state_changes - 1) // 2

    # Build per-position allowed nucleotides
    nucleotides = ['A', 'C', 'G', 'U']
    allowed = []
    for pos in range(half_length):
        if pos < target_gc_initial:
            allowed.append(['C', 'G'])
        elif pos == target_gc_initial and half_length > target_gc_initial:
            allowed.append(['A', 'U'])
        else:
            allowed.append(nucleotides)

    # Compute total restricted possibilities
    total_possibilities = 1
    for group in allowed:
        total_possibilities *= len(group)
    print(f"Starting candidate generation. Total possibilities (restricted): {total_possibilities:,}")

    # --- 1) Entropy filter ---
    valid_entropy_comps = {
        comp
        for comp in gen_compositions(half_length)
        if abs(compute_entropy(comp, half_length) - target_entropy) <= entropy_tol
    }
    print(f"Valid compositions after ENTROPY filter: {len(valid_entropy_comps)}")

    # --- 2) State-change filter ---
    valid_state_comps = {
        comp
        for comp in gen_compositions(half_length)
        if compute_state_change_range(comp, half_length)[0] 
           <= first_half_target_state_change 
           <= compute_state_change_range(comp, half_length)[1]
    }
    print(f"Valid compositions after STATE-CHANGE filter: {len(valid_state_comps)}")

    # --- 3) GC-initial filter ---
    valid_gc_comps = {
        comp
        for comp in gen_compositions(half_length)
        if (comp[1] + comp[2]) >= target_gc_initial
    }
    print(f"Valid compositions after GC-INITIAL filter: {len(valid_gc_comps)}")

    # --- Intersection of all three ---
    valid_comps = valid_entropy_comps & valid_state_comps & valid_gc_comps
    print(f"Compositions after ALL filters (intersection): {len(valid_comps)}")

    # Now iterate through the restricted ordering space
    candidate_count = 0
    for candidate in itertools.product(*allowed):
        candidate_count += 1
        if candidate_count % progress_freq == 0:
            print(f"Tested {candidate_count:,} candidates…")

        # quick composition check
        comp = tuple(candidate.count(nt) for nt in nucleotides)
        if comp not in valid_comps:
            continue

        # ordering‐dependent checks
        if state_change_count(candidate) != first_half_target_state_change:
            continue
        if initial_gc_count(candidate) != target_gc_initial:
            continue
        if abs(compute_entropy(comp, half_length) - target_entropy) > entropy_tol:
            continue

        # found a valid first half
        seq = "".join(candidate)
        return seq, candidate_count

    # nothing found
    print("Search completed. No valid candidate found.")
    return None, candidate_count


RANGES = {
    "Tamanho Loop": (3, 16),
    "A%_6_A_tract": (0, 1),
    "C%_6_A_tract": (0, 1),
    "U%_10_U_tract": (0, 1),
    "U%_6_U_tract": (0, 1),
    "A%_6_U_tract": (0, 1),
    "C%_U_tract": (0, 1),
    "Tamanho Hairpin sem Loop": (6, 49),
    "%GC_Loop": (0, 1),
    "Entropia_A_tract": (0, 2),
    "Entropia_U_tract": (0, 2),
    "Entropia_HP_S_Loop": (0, 2),
    "A_Tract_state-change": (0, 7),
    "U_Tract_state-change": (0, 11),
    "HP_S_Loop_state_change": (2, 34),
    "GC_Inicial_Hairpin": (0, 12),
}

def normalize_feat(val, feature_name):
    mn, mx = RANGES[feature_name]
    return (val - mn) / (mx - mn)


def extract_features(a_seq, u_seq, hairpin_seq, loop_seq):
    """
    Given the four parts as strings, compute the 16-feature vector **normalized** [0,1].
    """
    feats = {}
    # --- Loop features ---
    L_loop = len(loop_seq)
    feats["Tamanho Loop"] = L_loop
    feats["%GC_Loop"]   = loop_seq.count("G") / L_loop

    # --- A-tract features (8-mer, but we only percent-count the *last* 6) ---
    tail6_A = a_seq[-6:]                         # ← last 6 nt of the 8-mer
    feats["A%_6_A_tract"] = tail6_A.count("A") / 6
    feats["C%_6_A_tract"] = tail6_A.count("C") / 6

    # Entropy & state-changes on the *full* 8-mer
    countsA = (a_seq.count("A"), a_seq.count("C"), a_seq.count("G"), a_seq.count("U"))
    feats["Entropia_A_tract"]     = shannon_entropy(countsA)
    feats["A_Tract_state-change"] = state_change_count(list(a_seq))

    # --- U-tract features (12-mer) ---
    feats["U%_10_U_tract"] = u_seq[:10].count("U") / 10
    feats["U%_6_U_tract"]  = u_seq[:6].count("U")  / 6
    feats["A%_6_U_tract"]  = u_seq[:6].count("A")  / 6
    feats["C%_U_tract"]    = u_seq.count("C")      / 12
    countsU = (u_seq.count("A"), u_seq.count("C"), u_seq.count("G"), u_seq.count("U"))
    feats["Entropia_U_tract"]     = shannon_entropy(countsU)
    feats["U_Tract_state-change"] = state_change_count(list(u_seq))

    # --- Hairpin features ---
    half = len(hairpin_seq) // 2
    first_half = hairpin_seq[:half]
    feats["Tamanho Hairpin sem Loop"]   = half * 2
    feats["GC_Inicial_Hairpin"]         = initial_gc_count(first_half)
    countsH = (first_half.count("A"), first_half.count("C"),
               first_half.count("G"), first_half.count("U"))
    feats["Entropia_HP_S_Loop"]         = shannon_entropy(countsH)
    feats["HP_S_Loop_state_change"]     = state_change_count(list(first_half)) * 2 + 1

    # --- Build normalized feature vector ---
    feature_vector = []
    for name in desired_order:
        raw = feats[name]
        feature_vector.append(normalize_feat(raw, name))

    return np.array(feature_vector)

def compare_seq_vs_expected(a_seq, u_seq, hairpin_seq, loop_seq, df_expected):
    # recompute each feature in its raw (denormalized/integer) form:
    actual = {}
    # --- Loop ---
    L_loop = len(loop_seq)
    actual["Tamanho Loop"] = L_loop
    actual["%GC_Loop"]     = loop_seq.count("G")

    # --- A-tract (8mer, last 6) ---
    tail6 = a_seq[-6:]
    actual["A%_6_A_tract"] = tail6.count("A")
    actual["C%_6_A_tract"] = tail6.count("C")
    cnts = (a_seq.count("A"), a_seq.count("C"), a_seq.count("G"), a_seq.count("U"))
    actual["Entropia_A_tract"]     = shannon_entropy(cnts)
    actual["A_Tract_state-change"] = state_change_count(list(a_seq))

    # --- U-tract (12mer) ---
    actual["U%_10_U_tract"] = u_seq[:10].count("U")
    actual["U%_6_U_tract"]  = u_seq[:6].count("U")
    actual["A%_6_U_tract"]  = u_seq[:6].count("A")
    actual["C%_U_tract"]    = u_seq.count("C")
    cnts = (u_seq.count("A"), u_seq.count("C"), u_seq.count("G"), u_seq.count("U"))
    actual["Entropia_U_tract"]     = shannon_entropy(cnts)
    actual["U_Tract_state-change"] = state_change_count(list(u_seq))

    # --- Hairpin (first half) ---
    half = len(hairpin_seq)//2
    first = hairpin_seq[:half]
    actual["Tamanho Hairpin sem Loop"] = half*2
    actual["GC_Inicial_Hairpin"]       = initial_gc_count(first)
    cnts = (first.count("A"), first.count("C"), first.count("G"), first.count("U"))
    actual["Entropia_HP_S_Loop"]      = shannon_entropy(cnts)
    actual["HP_S_Loop_state_change"]  = state_change_count(list(first))*2 + 1

    # pull expected (integer/float) values from df_expected
    exp = df_expected.iloc[0].to_dict()

    # print side-by-side
    print("\nFeature                 | generated | expected |   Δ")
    print("------------------------+-----------+----------+--------")
    for feat in desired_order:
        g = actual[feat]
        e = exp[feat]
        delta = g - e
        print(f"{feat:24s} | {g:9.4f} | {e:8.4f} | {delta:+7.4f}")



#########################
# Main Function
#########################

def main():
    # We'll collect one record per hairpin size
    results = []

    # Pre‐built DE bounds
    bounds = [(0, 1)] * len(desired_order)

    # Loop over every even hairpin length from 6 to 48
    for hp_size in range(6, 49, 2):
        # 1) Set the hairpin‐size cap
        global max_hairpin_norm
        max_hairpin_norm = normalize_feat(hp_size, "Tamanho Hairpin sem Loop")

        print(f"\n=== Hairpin size: {hp_size} ===")
        # 2) Run DE to maximize predicted strength
        result = differential_evolution(
            cost_maximize,
            bounds,
            strategy="best1bin",
            maxiter=500,
            popsize=20,
            tol=1e-6,
            seed=0
        )

        # 3) Snap & re‐predict on discrete grid
        best_vec = result.x
        best_snapped = snap_to_valid_candidate(
            best_vec, loop_mapping, atract_mapping, utract_mapping, hairpin_mapping
        )
        max_strength = model.predict(best_snapped.reshape(1, -1))[0]
        print(f"Max strength @ hp={hp_size}: {max_strength:.4f}")

        # 4) Build real‐world features DataFrame for this snapped vector
        df_norm   = pd.DataFrame([best_snapped], columns=desired_order)
        df_denorm = denormalize_dataframe(df_norm)
        df_scaled = calculate_final_values(df_denorm)
        # only integer‐convert the count/length cols
        int_cols = [
            "Tamanho Loop","A%_6_A_tract","C%_6_A_tract","U%_10_U_tract",
            "U%_6_U_tract","A%_6_U_tract","C%_U_tract","Tamanho Hairpin sem Loop",
            "%GC_Loop","A_Tract_state-change","U_Tract_state-change",
            "HP_S_Loop_state_change","GC_Inicial_Hairpin"
        ]
        df_final = convert_columns_to_int(df_scaled, int_cols)
        # pull out a dict of real features
        real_features = df_final.iloc[0].to_dict()

        # 5) Generate the four sequence parts
        # A-tract
        a_row = df_final.iloc[0]
        ideal_a = generate_a_tract_by_target(
            float(a_row["Entropia_A_tract"]),
            int(a_row["A%_6_A_tract"]),
            int(a_row["C%_6_A_tract"]),
            int(a_row["A_Tract_state-change"])
        ) or ""

        # U-tract
        u_row = df_final.iloc[0]
        ideal_u, _ = generate_u_tract_by_distribution(
            float(u_row["Entropia_U_tract"]),
            int(u_row["U_Tract_state-change"]),
            int(u_row["U%_10_U_tract"]),
            int(u_row["U%_6_U_tract"]),
            int(u_row["A%_6_U_tract"]),
            int(u_row["C%_U_tract"]),
            entropy_tol=0.001,
            progress_freq=1000000
        )
        ideal_u = ideal_u or ""

        # Hairpin halves
        half_len = int(a_row["Tamanho Hairpin sem Loop"]) // 2
        hp_ent   = float(a_row["Entropia_HP_S_Loop"])
        hp_sc    = int(a_row["HP_S_Loop_state_change"])
        hp_gc    = int(a_row["GC_Inicial_Hairpin"])
        first_half, _ = generate_hairpin_first_half(
            half_len, hp_ent, hp_sc, hp_gc,
            entropy_tol=0.05,
            progress_freq=1000000
        )
        first_half = first_half or ""
        second_half = generate_second_half(first_half)

        # Loop
        loop_seq = generate_loop_sequence_by_features(df_final) or ""

        full_terminator = ideal_a + first_half + loop_seq + second_half + ideal_u

        # 6) Gather into a single record
        record = {
            "hairpin_length": hp_size,
            "max_strength":   max_strength,
            "A-tract":        ideal_a,
            "Hairpin_1st":    first_half,
            "Loop":           loop_seq,
            "Hairpin_2nd":    second_half,
            "U-tract":        ideal_u,
            "Terminator":     full_terminator
        }
        # merge on the real_features dict
        record.update(real_features)

        # 7) Export a one-row CSV for this hairpin size
        df_row = pd.DataFrame([record])
        df_row.to_csv(f"max_strength_hp_{hp_size}.csv", index=False)

        # collect for combined CSV
        results.append(record)

    # 8) Also write a combined file
    pd.DataFrame(results).to_csv("max_strength_all_hairpins.csv", index=False)
    print("\nAll done: CSVs written for each hairpin (6–48) and combined.")

if __name__ == "__main__":
    main()

