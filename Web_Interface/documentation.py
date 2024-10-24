import streamlit as st

def documentation():
    st.markdown(r"""
    # Documentation
    """)
    st.markdown(r"""
    ## Chemical Similarity Search
    ### Overview
    Chemical similarity search uses distance measures to assess the similarity or dissimilarity between compounds. These measures help evaluate how different two compounds are from each other, based on their binary fingerprint representation. They are important in the context of matching, searching, and classifying chemical information.
    ### Distance measures
    Different distance measures are used to compare the compounds, such as 
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(r"""
            #### Tanimoto
    Tanimoto distance is a simple yet powerful metric. It is defined as the ratio of the intersection of the sets to the union of the sets.

    $$
    T = \frac{|A \cap B|}{|A|  +  |B| - |A \cap B|}
    $$

        """)
    with col2:
        st.markdown(r"""
            #### Sørensen-Dice Coefficient
    The Dice distance, also known as the Dice coefficient. It is closely related to the Tanimoto coefficient and quantifies the degree of overlap between two sets of molecular features.

    $$
    D = \frac{2 \cdot |A \cap B|}{|A|  +  |B|}
    $$

        """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(r"""
            #### Cosine
            Cosine similarity is a common measure of similarity between two vectors. It is calculated by taking the cosine of the angle between the vectors.

            $$
            C = \frac{A \cdot B}{\|A\|  \cdot \|B\|}
            $$
                """)


    with col2:
        st.markdown(r""" 
            #### Tversky
        Tversky is a measure of dissimilarity between two molecular fingerprints.

        $$
        T_2 = \frac{|A \cap B|}{|A \cap B|  +  \alpha \cdot |A \backslash B|  + \beta \cdot |B \backslash A|}
        $$

        """)
