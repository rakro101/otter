import pandas as pd

if __name__ == "__main__":
    df_origin = pd.read_csv(
        "tables/Main_Table_2___Keystone_Species_OriginalSimCon_1-corr.csv", sep=";"
    )
    print(df_origin.columns)
    cols = ["Nodes", "Phylum", "Genus", "Species", "cluster_names", "relAbund", "cln"]
    new_cols = [
        "Nodes",
        "Phylum",
        "Genus",
        "Species",
        "Cluster",
        "rel. Abundance",
        "Closeness Centrality",
    ]
    cols_dict = dict(zip(cols, new_cols))
    df_origin[cols].rename(columns=cols_dict).to_latex(
        "tables/Main_Table_2___Keystone_Species_OriginalSimCon_Latex1-corr.tex", index=False
    )

    df_arc = pd.read_csv(
        "tables/Sup_Table_4___Keystone_Species_ArcticSimCon1-corr.csv", sep=";"
    )
    df_arc[cols].rename(columns=cols_dict).to_latex(
        "tables/Sup_Table_4___Keystone_Species_ArcticSimCon_Latex1-corr.tex", index=False
    )

    df_atl = pd.read_csv(
        "tables/Sup_Table_3___Keystone_Species_AtlanticSimCon1-corr.csv", sep=";"
    )
    df_atl[cols].rename(columns=cols_dict).to_latex(
        "tables/Sup_Table_3___Keystone_Species_AtlanticSimCon_Latex1-corr.tex", index=False
    )
