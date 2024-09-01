import pandas as pd
import streamlit as st

st.set_page_config(page_title="LatentSpaceEmbedding", page_icon="📈", layout="wide")

st.subheader("AI Embedding")
if st.button("Calculate Cluster Centroids distances",help="Calculate Cluster Centroids"):
    from gui_ai_umap_embedding import main_embeddings
    df_spec = st.session_state.df1
    st.write(df_spec)
    df_ccm =  st.session_state.df_pruned_ccmn
    meta =  st.session_state.df_enrich
    umap_3d, distance_matrix, pruned_distance_matrix = main_embeddings(df_spec, meta, df_ccm, hellinger=False,
                                                                       num_coefficients=st.session_state.FFT_COEFFS,
                                                                       save_pre_fig=st.session_state.PREFIX_FIG, save_pre_tab=st.session_state.PREFIX)
    plotcol1, plotcol2, plotcol3 = st.columns(3)
    st.write("Cluster Centroids distances Created")
    with plotcol1:
        st.write("Cluster Centroids")
        st.plotly_chart(umap_3d, use_container_width=True)
    with plotcol2:
        st.write("Unpruned Distance Matrix")
        st.pyplot(distance_matrix)
    with plotcol3:
        st.write("Pruned Distance Matrix")
        st.pyplot(pruned_distance_matrix)