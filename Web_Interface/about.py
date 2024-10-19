import streamlit as st
from lorem_text import lorem
import pandas as pd
from graphics.figures import *
from PIL import Image

df = pd.read_csv("./Web_Interface/data/cidals_user_view.csv")

text = lorem.paragraph()
meep = Image.open('./Web_Interface/media/Logo_MEEP.png')
bind = Image.open('./Web_Interface/media/logo_BIND.png')
ipt = Image.open('./Web_Interface/media/logo_IPT.png')
compound_distribution = Image.open('./Web_Interface/media/Distribution_of_compounds2.png')
all_logos = Image.open("./Web_Interface/media/all_logos.png")
def about():
    st.markdown(r"""
 # CidalsDB: An Open Resource for Anti-Pathogen Molecules

CidalsDB is an open resource on anti-pathogen molecules that provides a chemical similarity-based browsing function and an AI-based prediction tool for the 'cidal' effect of chemical compounds against pathogens. These include Leishmania parasites[^1^] and Coronaviruses (SARS-Cov, SARS-Cov-2, MERS)[^2^].

CidalsDB serves as an evolutive platform for democratized and no-code Computer-Aided Drug Discovery (CADD)[^3^]. We are committed to continuous improvement, actively incorporating additional pathogens, datasets, and AI predictive models.

[^1^]: Harigua-Souiai, E., Oualha, R., Souiai, O., Abdeljaoued-Tej, I. & Guizani, I. (2022). Applied machine learning toward drug discovery enhancement: Leishmaniases as a case study. Bioinforma. Biol. Insights 16, 11779322221090349.

[^2^]: Harigua-Souiai, E. et al. (2021). Deep learning algorithms achieved satisfactory predictions when trained on a novel collection of anticoronavirus molecules. Front. Genet. 12, 744170.

[^3^]: Harigua-Souiai, E., Masmoudi, O.,  Makni, S., Oualha, R., Abdelkrim, Y.Z., Hamdi, S., Souiai, O., Guizani, I. CidalsDB: An AI-empowered platform for anti-pathogen therapeutics research. Submitted.

    """)
    st.header("Datasets")
    st.write("""
    For now, we have datasets for two infectious diseases of interest within the **CidalsDB** database, that are accessible for the scientific community, namely *Leishmaniases* and *Coronaviruses*. For each disease, we performed an extensive search of the literature and retrieved data on molecules with validated anti-pathogen effects. We defined a data dictionary of published information related to the biological activity of the chemical compounds and used it to build the database. Then, we enriched the literature data with confirmatory screening datasets from PubChem. This led to consolidated sets of active and inactive molecules against Leishmania parasites and Coronaviruses. Additional infectious diseases will be considered to expand the database content.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://i.ibb.co/CmgtRyd/Distribution-of-compounds-leishmania-updated-02.png", width=450, caption="Molecules counts in the Leishmania dataset of CidalsDB")

    with col2:
        st.image("https://i.ibb.co/Mnd0nQ6/Distribution-of-categories-leishamniasi-updated-1.png", width=450, caption="Numbers of data entries in each experimental category in the Leishmania dataset of CidalsDB")

    colh, colg = st.columns(2)
    with colh:
        st.image("https://i.ibb.co/8ctdVWY/Distribution-of-compounds-Covid-updated-1.png", width=450, caption="Molecules counts in the Coronavirus dataset of CidalsDB")

    with colg:
        st.image("https://i.ibb.co/XzxzWz4/Distribution-of-categories-covid-updated-2.png", width=450, caption="Numbers of data entries in each experimental category in the Coronavirus dataset of CidalsDB")


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

    st.markdown(r"""## Workflow""")
    first_co, _ ,last_co = st.columns([0.5,0.1,0.4])
    with last_co:
        st.image("https://i.ibb.co/H71JzVd/search-2-drawio.png", output_format = "png", caption="Pipeline of Chemcial Search", width=300)
    with first_co:
        st.markdown(r"""
            The chemical search feature works by first having the user submit a SMILES query. This query is then converted into a fingerprint. Subsequently, the fingerprint is compared to the fingerprints of all the molecules in the dataset. Using one of the distance measures introduced earlier, it calculates the similarity between the query molecule and each of the molecules in the dataset. Finally, the N (<100) most similar molecules are returned to the user.
            """)
    st.markdown(r"""        
            ## Activity Prediction

            ### Encoding
            The process of data encoding involves transforming the SMILES representations from our molecule dataset into molecular fingerprints. This transformation allows us to extract features from each molecule and represent them in a suitable manner for machine learning algorithms.
            We opted to use RDkitfingerprints, which are binary vectors with a size we set to be 2048.

            ### Predictive models
            We trained and optimized ML and DL algorithms on the content of the CidalsDB, namely RF and MLP models on the Leishmania dataset and the GCN model on the Coronaviruses dataset. All technical details on the training, optimization and validation of the models can be found in the publication Harigua-Souia et al.[³](#cidalsdb-an-open-resource-for-anti-pathogen-molecules)""")


    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image("https://i.ibb.co/9Y8ppQ6/model-ml-drawio.png", output_format = "png", caption="Pipeline of Traning and Predictions", width=550)
    


    st.markdown(r"""
    ## Acknowledgement
    """)

    st.image(all_logos)
    
