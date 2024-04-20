import pandas as pd


def map_clu_season(x):
    dd = {"TS": "spring", "HS": "summer", "LW": "winter", "TA": "autumn"}
    return dd[x[2:]]


def tables_to_latex1_corr(in_path, out_path):
    df = pd.read_csv(in_path, sep=";", decimal=",")
    cols = [
        "Nodes",
        "Phylum",
        "Class",
        "Genus",
        "Species",
        "cluster_names",
        "relAbund",
        "cln",
    ]
    new_cols = [
        "Nodes",
        "Phylum",
        "Class",
        "Genus",
        "Species",
        "Cluster",
        "rel. Abundance",
        "Closeness Centrality",
    ]
    cols_dict = dict(zip(cols, new_cols))
    df.sort_values(["cluster_names", "relAbund"], inplace=True, ascending=False)

    df_temp = df[cols].rename(columns=cols_dict)
    df_temp["Nodes"] = df_temp["Nodes"].str.replace("_", "\_")
    df_temp["Phylum"] = df_temp["Phylum"].str.replace("_", "\_")
    df_temp["Genus"] = df_temp["Genus"].str.replace("_", "\_")
    df_temp["Species"] = df_temp["Species"].str.replace("_", "\_")
    df_temp["rel. Abundance"] = df_temp["rel. Abundance"].round(3)
    df_temp["Closeness Centrality"] = df_temp["Closeness Centrality"].round(3)
    # print(df_temp["Closeness Centrality"])
    df_temp["Species"] = df_temp["Species"].apply(lambda x: "\\textit{" + x + "}")
    # df_temp.value_counts("Cluster")
    df_temp2 = df_temp.copy()
    df_temp2["season"] = df_temp2["Cluster"].apply(lambda x: map_clu_season(x))
    print(df_temp2["season"].value_counts())
    print(df_temp2.value_counts(["season", "Phylum", "Class"]))
    latex_table = df_temp.to_latex(
        index=False, escape=False, decimal=".", float_format="%.3f"
    )
    latex_table = latex_table.replace("\n", "\n\\hline\n")
    with open(out_path, "w") as f:
        f.write(latex_table)

    return df


if __name__ == "__main__":
    df_origin = tables_to_latex1_corr(
        "tables/Main_Table_2___Keystone_Species_OriginalSimCon_1-corr.csv",
        "tables/Main_Table_2___Keystone_Species_OriginalSimCon_Latex1-corr.tex",
    )
    print("df_origin", df_origin.shape)
    df_arc = tables_to_latex1_corr(
        "tables/Sup_Table_4___Keystone_Species_ArcticSimCon1-corr.csv",
        "tables/Sup_Table_4___Keystone_Species_ArcticSimCon_Latex1-corr.tex",
    )
    print("df_arc", df_arc.shape)
    df_atl = tables_to_latex1_corr(
        "tables/Sup_Table_3___Keystone_Species_AtlanticSimCon1-corr.csv",
        "tables/Sup_Table_3___Keystone_Species_AtlanticSimCon_Latex1-corr.tex",
    )
    print("df_atl", df_atl.shape)
