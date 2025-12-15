
# constants for banzhaf_setup
const MAG_PERCENTILE = 0.95 # percentile threshold for magnitude-based filtering (filter_via_magnitude)

const MAX_INTERACTION_ORDER = 3 # maximum motif size to consider
const N_COALITION_PER_PT = 20 # how many random coalitions to generate per data point
const MIN_COALITION_SIZE = 2 # minimum size of coalition for random coalition 
const NUM_SAMPLES_PER_COALITION = 100 # how many samples to estimate banzhaf for a random coalition 
const Q_THRESHOLD = 1e-5 # FDR threshold for significant motifs