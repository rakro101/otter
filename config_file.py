""" Configuration file """

# Parameters
HELLENIGER_NORM = False
CON_METHOD = "Pearson_FFT"
CCMN_METHOD = "NMI"
FFT_COEFFS = 14
CON_TR = 0.70
CON_ALPHA = 0.05

CCMN_TR = 0.00
CCMN_ALPHA = "no"

LOUVAIN_RES = 1
CON_SYM = True
CCMN_SYM = False

# Sub p-value Parameters
NUM_PERMUTATIONS = 2
NUM_SAMPLES = 10
NUM_CORES = 1

# Input Files
ABUNDANCES_FILE = "tests/abundance.csv"
METADATA_FILE = "tests/environment_info.csv"
TAXA_FILE = "tests/taxa_info.csv"

PREFIX = "tests/tables"
RUN_ID = "PyTest"

# Output Files
CON_NETWORK_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_{CON_METHOD}__complete_network_table_{CON_TR}_{CON_ALPHA}.csv"
CON_META_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_{CON_METHOD}__complete_network_table_meta_{CON_TR}_{CON_ALPHA}.csv"

CCMN_NETWORK_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{CCMN_METHOD}__complete_network_table_{CCMN_TR}_{CCMN_ALPHA}.csv"
CCMN_META_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{CCMN_METHOD}__complete_network_table_meta_{CCMN_TR}_{CCMN_ALPHA}.csv"

CON_LOUVAIN_NETWORK_PATH = f"{PREFIX}/{RUN_ID}_Louvain_{LOUVAIN_RES}_{CON_METHOD}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}__complete_network_table_{CON_TR}_{CON_ALPHA}.csv"
CON_LOUVAIN_META_PATH = f"{PREFIX}/{RUN_ID}_Louvain_{LOUVAIN_RES}_{CON_METHOD}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}__complete_network_table_meta_{CON_TR}_{CON_ALPHA}.csv"

CCMN_CON_MAP_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_CCM_CON_MAP_Network.csv"

PVAL_CCMN_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_PV_CCM_CON_MAP_Network.csv"
PRUNED_PVAL_CCMN_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_Pruned_CCM_CON_MAP_Network.csv"
ENRICHED_META_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_Enriched_Hellinger_14_complete_network_table_meta_CON_CCM.csv"
