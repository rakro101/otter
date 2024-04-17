import pandas as pd

if __name__ == "__main__":
    df_meta = pd.read_csv("tables/Enriched_Paper_Meta.csv")
    # MaxMonthColor
    df_ccm = pd.read_csv(
        "tables/BeyondBlooms2024_Hellinger_True_14_Pruned_CCM_CON_MAP_Network.csv",
        sep=";",
    )
    print(df_ccm.shape)
    ccm_asv = list(set(df_ccm["from"].to_list() + df_ccm["to"].to_list()))
    print(len(ccm_asv))
    ccm_asv_dict = {asv: 1 for asv in ccm_asv}
    df_meta["CCM Significant"] = df_meta["Nodes"].map(ccm_asv_dict)
    df_meta["Color_New"] = df_meta.apply(
        lambda row: row["MaxMonthColor"] if row["CCM Significant"] == 1 else "#BBBBBB",
        axis=1,
    )
    print(df_meta["Color_New"].value_counts())
    df_meta.to_csv("tables/Sup_Figure_S11_Winterreset_meta_file.csv", index=False)
