from A_001_FP_create_con import create_con_network
from A_002_FP_create_ccm import create_ccmn_network
from A_003_FP_create_louvain import compute_louvain
from A_004_FP_create_map_ccm_to_con import ccmn_con_mapping
from A_005_FP_create_p_val_for_ccm import add_sub_pval_to_ccmn



def run_all():
    print("CON")
    create_con_network()
    print("CCM")
    create_ccmn_network()
    print("Louvain")
    compute_louvain()
    print("Mapping")
    ccmn_con_mapping()
    print("Signify")
    add_sub_pval_to_ccmn()
    print("Done")


if __name__ == "__main__":
    run_all()
