
# constants for banzhaf_setup
const MAG_PERCENTILE = 0.95 # percentile threshold for magnitude-based filtering (filter_via_magnitude)
const N_ROWS_THRESHOLD = 2 # number of rows each group (by :data_pt_index) must have to be kept in filtering_data_pts

const MAX_INTERACTION_ORDER = 3 # maximum motif size to consider
const N_COALITION_PER_PT = 20 # how many random coalitions to generate per data point
const MIN_COALITION_SIZE = 2 # minimum size of coalition for random coalition 
const NUM_SAMPLES_PER_COALITION = 100 # how many samples to estimate banzhaf for a random coalition 
const Q_THRESHOLD = 1e-5 # FDR threshold for significant motifs

const MAX_BG_DATA_PTs = 10000 # maximum background data points for significance testing
const MAX_BANZHAF_PER_GROUP = 5000 # maximum Banzhaf values per motif group