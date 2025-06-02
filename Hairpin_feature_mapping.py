import csv
import random
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── PARAMETERS ────────────────────────────────────────────────────────────────

NUCLEOTIDES    = ['A', 'C', 'G', 'U']
GC_BASES       = ['G', 'C']
COMP_MAP       = str.maketrans('ACGU', 'UGCA')

# Feature bounds & normalization constants
ENT_MIN, ENT_MAX = 0.0, 2.0     # entropy range for any hairpin (max log₂(4) = 2)
SC_MIN, SC_MAX   = 2,   34      # valid state-change ∈ [2, 34]
GC_MIN, GC_MAX   = 0,   12      # valid raw GC_initial ∈ [0, 12]

# Length normalization
L_MIN, L_MAX   = 6, 48
DENOM_LEN      = (L_MAX - L_MIN)  # = 42

# Monte Carlo settings
SAMPLES_PER_COMPOSITION = 300

# Adaptive‐sampling settings                                       <<< FIX #7
BATCH_SIZE = 5_000
REQUIRED_NO_NEW = 3

# Logging frequency
LOG_EVERY = 10_000  # log unique-feature count every LOG_EVERY draws per L

# Output files
FINAL_FEATURES_CSV    = 'features_MC_fullHairpin_L6_to_L48.csv'
FINAL_CONVERGENCE_CSV = 'convergence_log_fullHairpin_L6_to_L48.csv'


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def reverse_complement(seq: str) -> str:
    """Return the reverse-complement of an RNA sequence (A↔U, C↔G)."""
    return seq.translate(COMP_MAP)[::-1]


def compute_entropy(seq: str) -> float:
    """
    Compute Shannon entropy (base 2) of a nucleotide sequence.
    For any hairpin of length L, the maximum entropy over 4 letters ≤ 2.
    """
    counts = defaultdict(int)
    for nt in seq:
        counts[nt] += 1
    total = len(seq)

    # Guard against the empty‐string edge‐case
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return round(entropy, 4)


def count_state_changes(seq: str) -> int:
    """
    Count how many times adjacent nucleotides differ (i.e., state changes)
    along the given sequence of length L. Maximum = L - 1.
    """
    changes = 0
    for i in range(len(seq) - 1):
        if seq[i] != seq[i + 1]:
            changes += 1
    return changes


def count_gc_initial_raw(seq: str) -> int:
    """
    Count how many consecutive G or C appear from the 5' end until the first A or U.
    Returns an integer ≥ 0 (could be up to half_len, but we clamp > 12 later).
    """
    count = 0
    for nt in seq:
        if nt in GC_BASES:
            count += 1
        else:
            break
    return count


def enumerate_compositions(half_len: int):
    """
    Return a list of all (a_count, c_count, g_count, u_count) tuples
    such that a + c + g + u = half_len.
    """
    comps = []
    for a_count in range(half_len + 1):
        for c_count in range(half_len + 1 - a_count):
            for g_count in range(half_len + 1 - a_count - c_count):
                u_count = half_len - (a_count + c_count + g_count)
                comps.append((a_count, c_count, g_count, u_count))
    return comps


# Move make_jagged() outside the loop so it's not redefined each time      <<< FIX #4
def make_jagged(a_cnt, c_cnt, g_cnt, u_cnt):
    """
    “Most jagged” pattern for high state-change: alternate G/C as much as possible,
    then alternate A/U for leftovers.
    """
    half = []
    # Interleave G/C
    while g_cnt > 0 and c_cnt > 0:
        half.append('G')
        half.append('C')
        g_cnt -= 1
        c_cnt -= 1
    half.extend(['G'] * g_cnt)
    half.extend(['C'] * c_cnt)
    # Interleave A/U
    while a_cnt > 0 and u_cnt > 0:
        half.append('A')
        half.append('U')
        a_cnt -= 1
        u_cnt -= 1
    half.extend(['A'] * a_cnt)
    half.extend(['U'] * u_cnt)
    return ''.join(half)


def process_single_L(L: int):
    """
    Revised worker for length L (even 6..48), which:
      1) Seeds RNG per-L.
      2) For each composition of half_len = L/2:
         a) Generates “extremal seeds” (min/max SC, min/max ENT).
         b) Does adaptive sampling until that composition’s feature-space saturates.
      3) Writes unique (L_norm, ENT_norm, SC_norm, GC_init_n) → 'features_L{L}.csv'.
      4) Returns a list of convergence‐log entries.
    """
    # 1) Seed RNG so each L uses a different random sequence
    random.seed(42 + L)

    half_len = L // 2
    all_comps = enumerate_compositions(half_len)

    # Delete any old file for this L
    features_filename = f'features_L{L}.csv'
    try:
        os.remove(features_filename)
    except FileNotFoundError:
        pass

    with open(features_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=['L', 'L_norm', 'ENT_norm', 'SC_norm', 'GC_init_n']
        )
        writer.writeheader()

        # GLOBAL set of all feature‐tuples seen for this L
        seen_global = set()

        # This will collect per‐composition convergence logs
        convergence_log_L = []
        total_draws_global = 0
        next_global_log = LOG_EVERY                          # <<< FIX #1

        # ────────────────────────────────────────────────────────────────────
        # 2) For each composition, do seeds + adaptive sampling
        for (a_count, c_count, g_count, u_count) in all_comps:
            # Build the multiset for this composition
            half_base_list = (
                ['A'] * a_count +
                ['C'] * c_count +
                ['G'] * g_count +
                ['U'] * u_count
            )

            # LOCAL set for this composition’s feature tuples
            set_composition_seen = set()
            # draws_for_this_composition = 0  # <<< FIX #3: unused, remove if not needed
            consecutive_no_new = 0

            # 2a) “Extremal seeds” for this composition
            seeds = []

            # (i) completely sorted: A…A C…C G…G U…U
            sorted_half = ('A' * a_count) + ('C' * c_count) + ('G' * g_count) + ('U' * u_count)
            seeds.append(sorted_half)

            # (ii) reverse sorted: U…U G…G C…C A…A
            rev_sorted = ('U' * u_count) + ('G' * g_count) + ('C' * c_count) + ('A' * a_count)
            if rev_sorted not in seeds:                     # <<< FIX #5: skip duplicates
                seeds.append(rev_sorted)

            # (iii) “Most jagged” for high SC
            jagged_half = make_jagged(a_count, c_count, g_count, u_count)
            if jagged_half not in seeds:                    # <<< FIX #5
                seeds.append(jagged_half)

            # Process each “seed” exactly once
            for half_seq in seeds:
                full_seq = half_seq + reverse_complement(half_seq)

                # Clamp SC in [SC_MIN..SC_MAX]
                sc_full = count_state_changes(full_seq)
                if sc_full < SC_MIN:
                    sc_full = SC_MIN
                elif sc_full > SC_MAX:
                    sc_full = SC_MAX

                # Clamp GC_initial in [GC_MIN..GC_MAX]
                gc_init_raw = count_gc_initial_raw(half_seq)
                if gc_init_raw < GC_MIN:
                    gc_init = GC_MIN
                elif gc_init_raw > GC_MAX:
                    gc_init = GC_MAX
                else:
                    gc_init = gc_init_raw

                # Compute full‐hairpin entropy
                ent = compute_entropy(full_seq)
                ent_n = round((ent - ENT_MIN) / (ENT_MAX - ENT_MIN), 4)

                # Normalize SC, GC_initial, and L
                sc_n   = round((sc_full - SC_MIN) / (SC_MAX - SC_MIN), 4)
                gc_n   = round((gc_init - GC_MIN) / (GC_MAX - GC_MIN), 4)
                L_norm = round((L - L_MIN) / (DENOM_LEN), 4)

                key = (L_norm, ent_n, sc_n, gc_n)

                # If this key is new globally, write it and add to both sets
                if key not in seen_global:
                    seen_global.add(key)
                    writer.writerow({
                        'L': L,
                        'L_norm': L_norm,
                        'ENT_norm': ent_n,
                        'SC_norm': sc_n,
                        'GC_init_n': gc_n
                    })

                # Also record it in the composition‐local set
                set_composition_seen.add(key)

                # draws_for_this_composition += 1  # <<< FIX #3: unused
                total_draws_global += 1
                # Global logging every LOG_EVERY draws
                if total_draws_global >= next_global_log:
                    convergence_log_L.append({
                        'L': L,
                        'step': total_draws_global,
                        'unique_features': len(seen_global)
                    })
                    next_global_log += LOG_EVERY                   # <<< FIX #1

            # 2b) Adaptive random sampling
            while True:
                new_in_this_batch = 0
                for _ in range(BATCH_SIZE):
                    half_perm = random.sample(half_base_list, half_len)
                    half_seq  = ''.join(half_perm)
                    full_seq  = half_seq + reverse_complement(half_seq)

                    # Clamp SC
                    sc_full = count_state_changes(full_seq)
                    if sc_full < SC_MIN:
                        sc_full = SC_MIN
                    elif sc_full > SC_MAX:
                        sc_full = SC_MAX

                    # Clamp GC_initial
                    gc_init_raw = count_gc_initial_raw(half_seq)
                    if gc_init_raw < GC_MIN:
                        gc_init = GC_MIN
                    elif gc_init_raw > GC_MAX:
                        gc_init = GC_MAX
                    else:
                        gc_init = gc_init_raw

                    ent = compute_entropy(full_seq)
                    ent_n = round((ent - ENT_MIN) / (ENT_MAX - ENT_MIN), 4)
                    sc_n  = round((sc_full - SC_MIN) / (SC_MAX - SC_MIN), 4)
                    gc_n  = round((gc_init - GC_MIN) / (GC_MAX - GC_MIN), 4)
                    L_norm = round((L - L_MIN) / (DENOM_LEN), 4)

                    key = (L_norm, ent_n, sc_n, gc_n)

                    if key not in set_composition_seen:
                        # New for this composition
                        set_composition_seen.add(key)
                        # If new globally, write and add
                        if key not in seen_global:
                            seen_global.add(key)
                            writer.writerow({
                                'L': L,
                                'L_norm': L_norm,
                                'ENT_norm': ent_n,
                                'SC_norm': sc_n,
                                'GC_init_n': gc_n
                            })
                        new_in_this_batch += 1

                    total_draws_global += 1
                    if total_draws_global >= next_global_log:
                        convergence_log_L.append({
                            'L': L,
                            'step': total_draws_global,
                            'unique_features': len(seen_global)
                        })
                        next_global_log += LOG_EVERY               # <<< FIX #1

                # Check if no new bins appeared this batch
                if new_in_this_batch == 0:
                    consecutive_no_new += 1
                else:
                    consecutive_no_new = 0

                # If no new bins for REQUIRED_NO_NEW consecutive batches, stop
                if consecutive_no_new >= REQUIRED_NO_NEW:
                    break

            # End adaptive loop for this composition

        # End “for each composition”

        # 11) Guarantee at least one global log entry (for small L)
        if not convergence_log_L:
            convergence_log_L.append({
                'L': L,
                'step': total_draws_global,
                'unique_features': len(seen_global)
            })

    return convergence_log_L


def main():
    lengths = list(range(6, 49, 2))
    all_convergence_entries = []

    # Remove any stale per-L CSV files first
    for L in lengths:
        try:
            os.remove(f'features_L{L}.csv')
        except FileNotFoundError:
            pass

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_L, L): L for L in lengths}

        for future in as_completed(futures):
            L_done = futures[future]
            try:
                convergence_log_L = future.result()
                all_convergence_entries.extend(convergence_log_L)

                # Always pick the last entry (we guaranteed at least one)
                unique_count = convergence_log_L[-1]['unique_features']
                print(f"Finished L = {L_done}; Unique bins ≈ {unique_count}")
            except Exception as exc:
                print(f"L = {L_done} crashed: {exc}")

    # Write combined convergence log
    with open(FINAL_CONVERGENCE_CSV, 'w', newline='') as logfile:
        writer = csv.DictWriter(logfile, fieldnames=['L','step','unique_features'])
        writer.writeheader()
        for entry in all_convergence_entries:
            writer.writerow(entry)

    # Concatenate all features_L{L}.csv into one big CSV
    first_file = True
    with open(FINAL_FEATURES_CSV, 'w', newline='') as outfile:
        writer = None
        for L in lengths:
            fname = f'features_L{L}.csv'
            if not os.path.exists(fname):
                continue
            with open(fname, 'r', newline='') as infile:
                reader = csv.DictReader(infile)
                if first_file:
                    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    for row in reader:
                        writer.writerow(row)
                    first_file = False
                else:
                    for row in reader:
                        writer.writerow(row)

    # If nothing got written, create an empty file with only the header
    if first_file:
        with open(FINAL_FEATURES_CSV, 'w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=['L','L_norm','ENT_norm','SC_norm','GC_init_n'])
            writer.writeheader()

    print("All done.")
    print(f"Features → {FINAL_FEATURES_CSV}")
    print(f"Convergence log → {FINAL_CONVERGENCE_CSV}")


if __name__ == '__main__':
    main()
