import streamlit as st
import pandas as pd
import os

__author__ = (
    "Raphael Kronberg Department of MMBS, MatNat Faculty," " Heinrich-Heine-University"
)
__license__ = "MIT"
__version__ = "1.0.1"
__status__ = "Prototype: This progam/code can not be used as diagnostic tool."
__credits__ = (
    "Pls cite and refer to when using the code: Kronberg R.M.," "Beyond Blooms"
)


# Set page layout to wide
st.set_page_config(layout="wide")



# Title of the web app
head0, head00, head000 = st.columns([4, 1, 4])
with head0:
    st.title("Otter: Tool for Community Analysis")

with head00:
    try:
        st.image('img/otter_logo-remove-_bg.png', use_column_width=True)
    except:
        pass

with head000:
    try:
        st.image('img/awi-logo.png', caption='The development of this project is support by the Alfred Wegener Institute.', use_column_width=True)
    except:
        pass


col0, col00 = st.columns([3, 3])

########################################################################################################################
# Folder structure and Run name
########################################################################################################################
with col0:
    # Input fields for PREFIX and RUN_ID
    PREFIX = st.text_input("Fill in a Prefix", value="gui/tables")
    if PREFIX is not None:
        PREFIX_FIG = PREFIX.replace(PREFIX.split("/")[-1], "figures")
        if not os.path.exists(PREFIX):
            os.makedirs(PREFIX)
            st.write(f"Folder '{PREFIX}' created")
        if not os.path.exists(PREFIX_FIG):
            os.makedirs(PREFIX_FIG)
            st.write(f"Folder '{PREFIX_FIG}' created")

    RUN_ID = st.text_input("Fill in a Run ID", value="MyExperimentRun")



# Create sub-containers for file uploads
col1, col2, col3 = st.columns(3)
########################################################################################################################
# Upload the abundance, environment and taxa data
########################################################################################################################
# Upload three CSV files
with col1:
    st.title("Abundance")
    checkbox_value1 = st.checkbox('Transpose DF1', help="Swap rows and columns")
    # Dropdown to select separator
    separator1 = st.selectbox("Select Separator 1", [";", ",", "\t", "\s"],help="Choose the separator for your inputfile.")
    uploaded_file1 = st.file_uploader("Upload CSV File 1", type=["csv"],help="Uploaded CSV File")
    if uploaded_file1 is not None:
        df1 = pd.read_csv(uploaded_file1, sep=separator1, index_col=0)
        df1.index = df1.index.astype(str).str.replace(";", "_")
        #df1.drop_duplicates(inplace=
        df1 = df1[~df1.index.duplicated(keep='first')]
        if checkbox_value1:
            df1 = df1.T
        # df1 = df1.T.head(50).T
        num_events1 = df1.shape[0]
        num_asvs1 = df1.shape[1]
        st.write("Abundance: {} events, {} asvs".format(num_events1, num_asvs1))
        st.dataframe(df1)

with col3:
    st.title("Environment")
    checkbox_value2 = st.checkbox('Transpose DF2', help="Swap rows and columns")
    # Dropdown to select separator
    separator2 = st.selectbox("Select Separator 2", [";", ",", "\t", "\s"],help="Choose the separator for your inputfile.")
    uploaded_file2 = st.file_uploader("Upload CSV File 2", type=["csv"],help="Uploaded CSV File")
    if uploaded_file2 is not None:
        df2 = pd.read_csv(uploaded_file2, sep=separator2, index_col=0)
        df2.index = df2.index.astype(str).str.replace(";","_")
        df2 = df2[~df2.index.duplicated(keep='first')]
        #df2.drop_duplicates(inplace=True)
        if checkbox_value2:
            df2 = df2.T
        num_events2 = df2.shape[0]
        num_features2 = df2.shape[1]
        st.write(
            "Environment: {} events, {} features".format(num_events2, num_features2)
        )
        st.dataframe(df2)
    else:
        st.write("No Environment data")
        if uploaded_file1 is not None:
            df2 = df1.copy()
            df2["Environment"] = 1
            df2 = df2[["Environment"]]
            num_events2 = df2.shape[0]
            num_features2 = df2.shape[1]
            st.write(
                "Environment: {} events, {} features".format(num_events2, num_features2)
            )
            st.dataframe(df2)


with col2:
    st.title("Taxa")
    checkbox_value3 = st.checkbox('Transpose DF3', help="Swap rows and columns")
    # Dropdown to select separator
    separator3 = st.selectbox("Select Separator 3", [";", ",", "\t", "\s"],help="Choose the separator for your inputfile.")
    uploaded_file3 = st.file_uploader("Upload CSV File 3", type=["csv"],help="Uploaded CSV File")
    if uploaded_file3 is not None and uploaded_file1 is not None:
        df3 = pd.read_csv(uploaded_file3, sep=separator3, index_col=0)
        df3.index = df3.index.astype(str).str.replace(";", "_")
        df3 = df3[~df3.index.duplicated(keep='first')]
        df3 = df3[df3.index.isin(df1.columns)]
        #df3.drop_duplicates(inplace=True)
        if checkbox_value3:
            df3 = df3.T
        num_events3 = df3.shape[0]
        num_features3 = df3.shape[1]
        st.write("TaxaInfo: {} events, {} features".format(num_events3, num_features3))
        st.dataframe(df3)

########################################################################################################################
# Parameter Configs
########################################################################################################################

with col1:
    my_expander1 = st.expander(label="Expand CON Params")
    with my_expander1:
        st.title("CON Parameters")
        # Input fields
        HELLENIGER_NORM = st.checkbox("Hellinger Norm", value=False, help="Use Hellinger Normalisation")
        CON_SYM = st.checkbox("Connectivity Symmetry", value=True, help="Remove directed egdes")
        CON_METHOD = st.selectbox("Connectivity Method", ["Pearson_FFT", "Other"],help="Select a co-occurrence measurement")
        FFT_COEFFS = st.number_input("FFT Coefficients", value=14, min_value=1,help="Select the number of FFT coefficients")
        CON_TR = st.slider(
            "Connectivity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.01,
            help="Select the connectivity threshold for edge weights, edges below this value will be removed."
        )
        CON_ALPHA = st.slider(
            "Connectivity Alpha", min_value=0.0, max_value=1.0, value=0.05, step=0.01,help="Select the connectivity threshold the pvalue, edges below this value will be removed."
        )

with col2:
    my_expander2 = st.expander(label="Expand CCM Params")
    with my_expander2:
        st.title("CCM Parameters")
        CCMN_SYM = st.checkbox("CCMN Symmetry", value=False, help="False if methods is non symmetric")
        CCMN_METHOD = st.selectbox("CCMN Method", ["NMI", "Other"],help="Select the CCMN Method")
        CCMN_TR = st.slider(
            "CCMN Threshold", min_value=0.0, max_value=1.0, value=0.00, step=0.01, help="Select the cut-off value for the edge weighs, edge below will be removed.",
        )
        CCMN_ALPHA = st.radio("CCMN Alpha", ["yes", "no"], help="Select the cut-off of the p_value. Not implemented for CCM using NMI.")
        LOUVAIN_RES = st.number_input("Louvain Resolution", value=1, min_value=1, help="Select the Louvain resolution.")

with col3:
    my_expander3 = st.expander(label="Expand Permu Params")
    with my_expander3:
        # Sub p-value Parameters
        st.title("Permutation Parameters")
        CALC_CENTRALITIES = st.selectbox("Centralities", [False, True], help="If checked, theCentralities will be calculated.")
        NUM_PERMUTATIONS = st.number_input(
            "Number of Permutations", value=100, min_value=1, help="Number of permutations to calculate the permutation significance value."
        )
        NUM_SAMPLES = st.number_input("Number of Samples", value=100, min_value=10, help="Number of samples to calculate the permutation significance")
        NUM_CORES = st.number_input("Number of Cores", value=10, min_value=1, help="Number of cores on your machine to calculate the permutation significance value.")

########################################################################################################################
# Output files
########################################################################################################################

# Output Files
CON_NETWORK_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_{CON_METHOD}__complete_network_table_{CON_TR}_{CON_ALPHA}.csv"
CON_META_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_{CON_METHOD}__complete_network_table_meta_{CON_TR}_{CON_ALPHA}.csv"

CCMN_NETWORK_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{CCMN_METHOD}__complete_network_table_{CCMN_TR}_{CCMN_ALPHA}.csv"
CCMN_META_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{CCMN_METHOD}__complete_network_table_meta_{CCMN_TR}_{CCMN_ALPHA}.csv"

CON_LOUVAIN_NETWORK_PATH = f"{PREFIX}/{RUN_ID}_Louvain_{LOUVAIN_RES}_{CON_METHOD}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}__complete_network_table_{CON_TR}_{CON_ALPHA}.csv"
CON_LOUVAIN_META_PATH = f"{PREFIX}/{RUN_ID}_Louvain_{LOUVAIN_RES}_{CON_METHOD}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}__complete_network_table_meta_{CON_TR}_{CON_ALPHA}.csv"

CCMN_CON_MAP_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_CCM_CON_MAP_Network.csv"

PVAL_CCMN_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_Samples_{NUM_SAMPLES}_Permus_{NUM_PERMUTATIONS}_PV_CCM_CON_MAP_Network.csv"
PRUNED_PVAL_CCMN_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_Samples_{NUM_SAMPLES}_Permus_{NUM_PERMUTATIONS}_Pruned_CCM_CON_MAP_Network.csv"
ENRICHED_META_PATH = f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_Samples_{NUM_SAMPLES}_Permus_{NUM_PERMUTATIONS}_Enriched_Hellinger_14_complete_network_table_meta_CON_CCM.csv"

RANDOM_PVAL_CCMN_PATH = (
    f"{PREFIX}/{RUN_ID}_Hellinger_{HELLENIGER_NORM}_{FFT_COEFFS}_Samples_{NUM_SAMPLES}_Permus_{NUM_PERMUTATIONS}_RANDOM_Network.csv"
)


########################################################################################################################
# Check Input  files
########################################################################################################################

with col00:
    st.subheader("Check")

    # Check if dataframes exist
    def check_dataframes_exist(df1, df2, df3):
        if all([df1 is not None, df2 is not None, df3 is not None]):
            st.write("All dataframes are present ✅")
            # Check if df1 index is also found in df2 index
            if df1.index.isin(df2.index).all():
                st.write("Abundance (df1) index is found in Environment (df2) index ✅")
            else:
                st.write(
                    "Abundance (df1) index is not found in Environment (df2) index ❌"
                )
            # Check if df1 columns are rows in df3
            if set(df1.columns).issubset(df3.index) and set(df3.index).issubset(
                df1.columns
            ):
                st.write("Abundance (df1) columns are rows in Taxa (df3) ✅")
            else:
                st.write("Abundance (df1) columns are not rows in Taxa (df3) ❌")
                st.write(f"df1 cols:{len(df1.columns)}  df3 index: {len(df3.index)}")
        else:
            st.write("Some dataframes are missing ❌")

    # Display the check result
    if st.button("Check input data", help="Check if you input data matches our criteria"):
        try:
            check_dataframes_exist(df1, df2, df3)
        except:
            st.write("Some dataframes are missing ❌")


########################################################################################################################
# Output Network Files Buttons and Statistics
########################################################################################################################

# Create sub-containers for file uploads
col4, col5, col6, col7, col8, col9 = st.columns(6)

with col4:
    st.subheader("CON")
    if st.button("Create Co-occurrence Network", help="Create Co-occurrence Network"):
        print("Creating Co-occurrence Network")
        from gui_create_con import create_con_network

        with st.spinner("Create CON..."):
            create_con_network(
                HELLENIGER_NORM,
                CON_METHOD,
                FFT_COEFFS,
                df2,
                df1,
                df3,
                CON_TR,
                CON_ALPHA,
                CON_SYM,
                CON_NETWORK_PATH,
                CON_META_PATH,
            )
        st.write("Con Created")
    try:
        df_con = pd.read_csv(CON_NETWORK_PATH, sep=";")
        df_con.drop(df_con.columns[4:], axis=1, inplace=True)
        num_events_con = df_con.shape[0]
        num_features_con = len(
            set(set(df_con["from"].to_list()).union(df_con["to"].to_list()))
        )
        st.write(
            "Con Info: {} edges, {} nodes".format(num_events_con, num_features_con)
        )
        st.dataframe(df_con.describe())
        st.download_button(
            label="Download CON CSV",
            data=df_con.to_csv(index=False, sep=","),  # Convert DataFrame to CSV format
            file_name=f"{PREFIX}_Raw_CON_File.csv",  # Specify the desired file name
            mime="text/csv",  # Set the MIME type
            help = "Download the file."
        )
    except:
        st.write("Con not calculated")


with col5:
    st.subheader("CCMN")
    if st.button("Create CCM Network", help="Create CCM Network"):
        print("Creating CCM Network")
        from gui_create_ccm import create_ccmn_network

        with st.spinner("Create CCMN..."):
            create_ccmn_network(
                HELLENIGER_NORM,
                CCMN_METHOD,
                df2,
                df1,
                df3,
                CCMN_SYM,
                CCMN_NETWORK_PATH,
                CCMN_META_PATH,
            )
        st.write("CCM Created")
    try:
        df_ccmn = pd.read_csv(CCMN_NETWORK_PATH, sep=";")
        df_ccmn.drop(df_con.columns[3:], axis=1, inplace=True)
        num_events_ccmn = df_ccmn.shape[0]
        num_features_ccmn = len(
            set(set(df_ccmn["from"].to_list()).union(df_ccmn["to"].to_list()))
        )
        st.write(
            "CCMN Info: {} edges, {} nodes".format(num_events_ccmn, num_features_ccmn)
        )
        st.dataframe(df_ccmn.describe())
        st.download_button(
            label="Download CCMN CSV",
            data=df_ccmn.to_csv(
                index=False, sep=","
            ),  # Convert DataFrame to CSV format
            file_name=f"{PREFIX}_Raw_CCMN_File.csv",  # Specify the desired file name
            mime="text/csv",  # Set the MIME type
            help="Download the file."
        )
    except:
        st.write("CCM not calculated")

with col6:
    st.subheader("Louvain")
    if st.button("Create Louvain Cluster Network", help="Create Louvain Clusters."):
        print("Creating Co-occurrence Network")
        from gui_create_louvain import compute_louvain

        with st.spinner("Create Louvain Meta..."):
            compute_louvain(
                CON_TR,
                CON_NETWORK_PATH,
                CON_META_PATH,
                LOUVAIN_RES,
                CON_LOUVAIN_NETWORK_PATH,
                CON_LOUVAIN_META_PATH,
            )
    try:
        df_louvain = pd.read_csv(CON_LOUVAIN_META_PATH, sep=";")
        print(df_louvain.columns)
        num_l_classes = len(df_louvain["LouvainLabelD"].unique())
        st.write("Number of Louvain Clusters: {}".format(num_l_classes))
        st.dataframe(df_louvain.describe())
        st.download_button(
            label="Download Louvain CSV",
            data=df_louvain.to_csv(
                index=False, sep=","
            ),  # Convert DataFrame to CSV format
            file_name=f"{PREFIX}_Louvain_Meta.csv",  # Specify the desired file name
            mime="text/csv",  # Set the MIME type
            help="Download the file."
        )
    except:
        st.write("Louvain not calculated")

with col7:
    st.subheader("Mapping")
    if st.button("Create Mapping Network", help="Create a mapping network"):
        print("Create Mapping Network")
        from gui_create_mapping import ccmn_con_mapping

        with st.spinner("Create Mapping ..."):
            ccmn_con_mapping(
                CCMN_CON_MAP_PATH,
                CON_LOUVAIN_META_PATH,
                CON_LOUVAIN_NETWORK_PATH,
                CCMN_NETWORK_PATH,
            )
        st.write("Mapping finished")
    try:
        st.write("Mapping finished")
        df_map = pd.read_csv(CCMN_CON_MAP_PATH, sep=";")
        df_map.drop(columns="Unnamed: 0", inplace=True)
        st.dataframe(df_map.describe())
        st.download_button(
            label="Download Map CSV",
            data=df_map.to_csv(index=False, sep=","),  # Convert DataFrame to CSV format
            file_name=f"{PREFIX}_Map.csv",  # Specify the desired file name
            mime="text/csv",  # Set the MIME type
            help="Download the file."
        )
    except:
        st.write("Mapping not calculated")

with col8:
    st.subheader("Permutation")
    if st.button("Create Permutation Pruning Network", help="Create the permutation network"):
        print("Create Permutation Pruning Network")
        from gui_create_permu_ccmn import add_sub_pval_to_ccmn

        with st.spinner("Create Pruned CCMN ..."):
            add_sub_pval_to_ccmn(
                df1,
                CCMN_CON_MAP_PATH,
                CON_LOUVAIN_META_PATH,
                CON_LOUVAIN_NETWORK_PATH,
                NUM_PERMUTATIONS,
                NUM_SAMPLES,
                NUM_CORES,
                df2,
                PRUNED_PVAL_CCMN_PATH,
                PVAL_CCMN_PATH,
                ENRICHED_META_PATH,
                RANDOM_PVAL_CCMN_PATH,
                CALC_CENTRALITIES,
            )
        st.write("CCMN finished")
    try:
        df_pruned_ccmn = pd.read_csv(PRUNED_PVAL_CCMN_PATH, sep=";")
        df_pruned_ccmn.drop(columns="Unnamed: 0", inplace=True)
        num_events_pruned_ccmn = df_pruned_ccmn.shape[0]
        num_features_pruned_ccmn = len(
            set(
                set(df_pruned_ccmn["from"].to_list()).union(
                    df_pruned_ccmn["to"].to_list()
                )
            )
        )
        st.write(
            "CCMN pruned Info: {} edges, {} nodes".format(
                num_events_pruned_ccmn, num_features_pruned_ccmn
            )
        )
        st.dataframe(df_pruned_ccmn.describe())
        st.download_button(
            label="Download Pruned CCMN CSV",
            data=df_pruned_ccmn.to_csv(
                index=False, sep=","
            ),  # Convert DataFrame to CSV format
            file_name=f"{PREFIX}_Pruned_CCMN_Samples_{NUM_SAMPLES}_Permus_{NUM_PERMUTATIONS}.csv",  # Specify the desired file name
            mime="text/csv",  # Set the MIME type
            help="Download the file."
        )
        df_enrich = pd.read_csv(ENRICHED_META_PATH, sep=",")
        df_enrich.drop(columns="Unnamed: 0", inplace=True)
        st.write("ENRICH")
        st.dataframe(df_enrich.describe())
        st.download_button(
            label="Download Enriched CSV",
            data=df_enrich.to_csv(
                index=False, sep=","
            ),  # Convert DataFrame to CSV format
            file_name=f"{PREFIX}_Enriched_Meta_File_Samples_{NUM_SAMPLES}_Permus_{NUM_PERMUTATIONS}_.csv",  # Specify the desired file name
            mime="text/csv",  # Set the MIME type
            help="Download the file."
        )
        df_non_pruned = pd.read_csv(PVAL_CCMN_PATH, sep=";")
        df_non_pruned.drop(columns="Unnamed: 0", inplace=True)
        st.write("Non_Pruned")
        st.dataframe(df_non_pruned.describe())
        st.download_button(
            label="Download NonPruned CSV",
            data=df_non_pruned.to_csv(
                index=False, sep=","
            ),  # Convert DataFrame to CSV format
            file_name=f"{PREFIX}_Non_PrunedP_CCMN_File_Samples_{NUM_SAMPLES}_Permus_{NUM_PERMUTATIONS}_.csv",  # Specify the desired file name
            mime="text/csv",  # Set the MIME type
            help="Download the file."
        )
    except:
        st.write("Pruning not calculated")

with col9:
    st.subheader("Run all")
    if st.button("Run all",help="Run all Calculations: Step by step."):
        with st.spinner("Create All Nets..."):
            from gui_create_con import create_con_network
            from gui_create_ccm import create_ccmn_network
            from gui_create_mapping import ccmn_con_mapping
            from gui_create_louvain import compute_louvain
            from gui_create_permu_ccmn import add_sub_pval_to_ccmn

            create_con_network(
                HELLENIGER_NORM,
                CON_METHOD,
                FFT_COEFFS,
                df2,
                df1,
                df3,
                CON_TR,
                CON_ALPHA,
                CON_SYM,
                CON_NETWORK_PATH,
                CON_META_PATH,
            )
            st.write("Con Created")

            create_ccmn_network(
                HELLENIGER_NORM,
                CCMN_METHOD,
                df2,
                df1,
                df3,
                CCMN_SYM,
                CCMN_NETWORK_PATH,
                CCMN_META_PATH,
            )
            st.write("CCM Created")

            compute_louvain(
                CON_TR,
                CON_NETWORK_PATH,
                CON_META_PATH,
                LOUVAIN_RES,
                CON_LOUVAIN_NETWORK_PATH,
                CON_LOUVAIN_META_PATH,
            )
            st.write("Louvain Created")


            ccmn_con_mapping(
                CCMN_CON_MAP_PATH,
                CON_LOUVAIN_META_PATH,
                CON_LOUVAIN_NETWORK_PATH,
                CCMN_NETWORK_PATH,
            )
            st.write("Mapping Created")

            add_sub_pval_to_ccmn(
                df1,
                CCMN_CON_MAP_PATH,
                CON_LOUVAIN_META_PATH,
                CON_LOUVAIN_NETWORK_PATH,
                NUM_PERMUTATIONS,
                NUM_SAMPLES,
                NUM_CORES,
                df2,
                PRUNED_PVAL_CCMN_PATH,
                PVAL_CCMN_PATH,
                ENRICHED_META_PATH,
                RANDOM_PVAL_CCMN_PATH,
                CALC_CENTRALITIES,
            )
            st.write("Pruning Created")

st.subheader("AI Embedding")
if st.button("Calculate Cluster Centroids distances",help="Calculate Cluster Centroids"):
    from gui_ai_umap_embedding import main_embeddings
    df_spec = df1
    df_ccm = pd.read_csv(PRUNED_PVAL_CCMN_PATH, sep=";")
    meta = pd.read_csv(ENRICHED_META_PATH, sep=",")
    umap_3d, distance_matrix = main_embeddings(df_spec, meta, df_ccm, hellinger=False, num_coefficients = FFT_COEFFS,save_pre_fig =PREFIX_FIG, save_pre_tab =PREFIX)
    plotcol1, plotcol2  = st.columns(2)
    st.write("Cluster Centroids distances Created")
    with plotcol1:
        st.write("Cluster Centroids")
        st.plotly_chart(umap_3d, use_container_width=True)
    with plotcol2:
        st.write("Unpruned Distance Matrix")
        st.pyplot(distance_matrix)



