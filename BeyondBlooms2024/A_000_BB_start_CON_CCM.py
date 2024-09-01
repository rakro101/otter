from A_001_BB_create_con import create_con_network
from A_002_BB_create_ccm import create_ccmn_network
from A_003_BB_create_louvain import compute_louvain
from A_004_BB_create_anti_map_ccm_to_con import \
    ccmn_con_mapping as ccmn_anti_mapping
from A_004_BB_create_map_ccm_to_con import ccmn_con_mapping
from A_005_BB_create_p_val_for_ccm import add_sub_pval_to_ccmn
from MV_001_BB_create_con import create_con_network as mv_create_con_network
from MV_005_BB_create_p_val_for_con import \
    add_sub_pval_to_ccmn as mv_add_sub_pval_to_ccmn


def run_all():
    print("CON")
    create_con_network()
    print("CCM")
    create_ccmn_network()
    print("Louvain")
    compute_louvain()
    print("Mapping")
    ccmn_con_mapping()
    print("Anti-mapping")
    ccmn_anti_mapping()
    print("Signify")
    add_sub_pval_to_ccmn()
    print("Done")

    print("MV")
    mv_create_con_network()
    print("MV 2")
    mv_add_sub_pval_to_ccmn()


if __name__ == "__main__":
    run_all()
