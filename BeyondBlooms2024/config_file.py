""" BB Configuration file """

# Parameters
HELLENIGER_NORM = True
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
NUM_PERMUTATIONS = 1000
NUM_SAMPLES = 1000
NUM_CORES = 8

# Input Files
ABUNDANCES_FILE = "./data/F4_euk_abundance_table.csv"
METADATA_FILE = "./data/F4_euk_meta_data.csv"
TAXA_FILE = "./data/F4_euk_taxa_info_table.csv"

ENRICH = "./tables/Enriched_Paper_Meta.csv"
PREFIX = "tables"
RUN_ID = "BeyondBlooms2024"

# Output Files
CON_NETWORK_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_{CON_METHOD}__complete_network_table_{CON_TR}_{CON_ALPHA}.csv"
CON_META_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_{CON_METHOD}__complete_network_table_meta_{CON_TR}_{CON_ALPHA}.csv"

MV_CON_NETWORK_PATH = f"{PREFIX}/{RUN_ID}_MV_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_{CON_METHOD}__complete_network_table_{0}_{None}.csv"
MV_CON_META_PATH = f"{PREFIX}/{RUN_ID}_MV_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_{CON_METHOD}__complete_network_table_meta_{0}_{None}.csv"

CCMN_NETWORK_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{CCMN_METHOD}__complete_network_table_{CCMN_TR}_{CCMN_ALPHA}.csv"
CCMN_META_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{CCMN_METHOD}__complete_network_table_meta_{CCMN_TR}_{CCMN_ALPHA}.csv"

CON_LOUVAIN_NETWORK_PATH = f"{PREFIX}/{RUN_ID}_Louvain_{LOUVAIN_RES}_{CON_METHOD}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}__complete_network_table_{CON_TR}_{CON_ALPHA}.csv"
CON_LOUVAIN_META_PATH = f"{PREFIX}/{RUN_ID}_Louvain_{LOUVAIN_RES}_{CON_METHOD}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}__complete_network_table_meta_{CON_TR}_{CON_ALPHA}.csv"

CCMN_CON_MAP_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_CCM_CON_MAP_Network.csv"
ANTI_CCMN_CON_MAP_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_ANTI_CCM_CON_MAP_Network.csv"

PVAL_CCMN_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_PV_CCM_CON_MAP_Network.csv"
PRUNED_PVAL_CCMN_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_Pruned_CCM_CON_MAP_Network.csv"

MV_PVAL_CCMN_PATH = f"{PREFIX}/{RUN_ID}_MV_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_PV_Permutaion_CON_Network.csv"
MV_PRUNED_PVAL_CCMN_PATH = f"{PREFIX}/{RUN_ID}_MV_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_Pruned_Permutaion_CON_Network.csv"

ENRICHED_META_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_Enriched_Hellinger_complete_network_table_meta_CON_CCM.csv"
RANDOM_PVAL_CCMN_PATH = (
    f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_RANDOM_Network.csv"
)
