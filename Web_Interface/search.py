
import streamlit as st
from stmol import showmol
import py3Dmol
# from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import AllChem
import pubchempy
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from scipy.spatial import distance
import pubchempy as pcp
import pandas as pd
from scipy.spatial.distance import dice, cosine
import re


captcha_codes = []
captcha_codez = []


def pubchem_id_to_smiles(pubchem_id):
    try:
        compound = pcp.get_compounds(pubchem_id)[0]
        smiles = compound.canonical_smiles
        return smiles
    except (IndexError, pcp.PubChemHTTPError) as e:
        print(f"Error retrieving SMILES for PubChem ID {pubchem_id}: {str(e)}")
        return None

def calculate_distance(smiles, fingerprint_array):
    molecule = Chem.MolFromSmiles(smiles)
    fingerprint = np.array(AllChem.RDKFingerprint(molecule, fpSize=1024))
    distances = distance.cdist(fingerprint.reshape(-1,1), fingerprint_array.reshape(-1,1), 'hamming')
    
    return distances.mean()


def passes_lipinski_rule(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    molecular_weight = Descriptors.MolWt(molecule)
    logp = Descriptors.MolLogP(molecule)
    hbd = Descriptors.NumHDonors(molecule)
    hba = Descriptors.NumHAcceptors(molecule)
    
    if molecular_weight > 500 or logp > 5 or hbd > 5 or hba > 10:
        return False
    else:
        return True


def makeblock(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mblock = Chem.MolToMolBlock(mol)
    return mblock

def render_mol(xyz):
    xyzview = py3Dmol.view()#(width=400,height=400)
    xyzview.addModel(xyz,'mol')
    xyzview.setStyle({'stick':{}})
    xyzview.setBackgroundColor('white')
    xyzview.zoomTo()
    showmol(xyzview,height=400,width=800)

def generate_name(smiles):
    compounds = pubchempy.get_compounds(smiles, namespace='smiles')
    match = compounds[0]
    return match.iupac_name


def highlight_active(val):
    if val == 'Active':
        return 'color: green; font-weight: 800'
    else:
        return ''

def calculate_similarity(query_smiles, df_smiles, fingerprint_array, distance_metric='tanimoto'):
    query_molecule = Chem.MolFromSmiles(query_smiles)
    query_fingerprint = AllChem.GetMorganFingerprintAsBitVect(query_molecule, 2)
    query_array = np.array(query_fingerprint, dtype=bool)
    
    similarities = []
    
    for i, df_smile in enumerate(df_smiles):
        df_molecule = Chem.MolFromSmiles(df_smile)
        df_fingerprint = fingerprint_array[i]
        
        if distance_metric == 'Tanimoto':
            similarity = np.sum(query_array & df_fingerprint) / np.sum(query_array | df_fingerprint)
        elif distance_metric == 'Sørensen–Dice':
            similarity = 1 - dice(query_array, df_fingerprint)
        elif distance_metric == 'Cosine':
            similarity = 1 - cosine(query_array, df_fingerprint)
        elif distance_metric == 'Tversky':
            alpha = 0.3 
            tversky_similarity = np.sum(query_array & df_fingerprint) / (
                np.sum(query_array & df_fingerprint) + alpha * np.sum(query_array & ~df_fingerprint) +
                (1 - alpha) * np.sum(~query_array & df_fingerprint)
            )
            similarity = tversky_similarity
        elif distance_metric == 'Average':
            alpha = 0.3 
            tanimoto_similarity = np.sum(query_array & df_fingerprint) / np.sum(query_array | df_fingerprint)
            dice_similarity = 1 - dice(query_array, df_fingerprint)
            cosine_similarity = 1 - cosine(query_array, df_fingerprint)
            tversky_similarity = np.sum(query_array & df_fingerprint) / (
                np.sum(query_array & df_fingerprint) + alpha * np.sum(query_array & ~df_fingerprint) +
                (1 - alpha) * np.sum(~query_array & df_fingerprint)
            )
            similarity = np.mean([tanimoto_similarity, dice_similarity, cosine_similarity, tversky_similarity])
        else:
            raise ValueError("Invalid distance_metric specified")
        
        similarities.append(similarity)
    
    return similarities


def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$"
    return re.match(pattern, email)

def search():

    st.markdown("<h1 style='font-size:2rem;'>Search</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:1.2rem;'>Please insert a valid SMILE</p>",
        unsafe_allow_html=True
    )

    """
    ### Search By Smiles
    """

    np_data = np.load("./Web_Interface/data/data.npy")
    df_data = pd.read_csv("./Web_Interface/data/cidals.csv", dtype={'PUBCHEM_CID': 'Int32'})

    tabs_css = """
    <style>
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 20px;
    }
    </style>
    """

    st.write(tabs_css, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Molecule SMILE", 'PUBCHEM ID'])
    with tab1:
        smile = st.text_input(label='Molecule SMILE', placeholder='COC1=C(C=C(C=C1)F)C(=O)C2CCCN(C2)CC3=CC4=C(C=C3)OCCO4')
        st.markdown(
                """
                <style>
                input[type="text"] {
                    font-size: 20px !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        col1, col2, col3, col4 = st.columns([0.4, 0.2, 0.2, 0.2])
        with col1:
            N = st.slider("Choose the number of closest molecules to display", 1, 100, 10, key=3)
            st.markdown(
                """
                <style>
                [data-testid="stMarkdownContainer"] p {
                    font-size: 20px !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        with col2:
            show_active_only = st.selectbox("Show Biological Activity", ['Active', 'Inactive', 'All'], key=10)
        with col3:
            distance = 'Sørensen–Dice'
            distance = st.selectbox('Distance Measure',('Sørensen–Dice', 'Tanimoto', 'Cosine', 'Tversky', 'Average'), key = 44)
        with col4:
            patho = "All"
            patho = st.selectbox('Pathogens',('Coronaviruses', 'Leishmaniases', 'All'), key = 4143)
        if not smile:
            pass
        else:
            try:
                with st.spinner("Please wait"):
                    try:
                        name = generate_name(smile)
                        st.caption(name)
                    except:
                        print("cant generate name")
                df_smiles = df_data['SMILES'].tolist()
                similarities = calculate_similarity(smile, df_smiles, np_data, distance_metric = distance)
                df_data['Chemical Distance Similarity'] = similarities
                df_data.sort_values(by='Chemical Distance Similarity', inplace=True, ascending=False)
                filtered_df = df_data.head(N)
                filtered_df.insert(0, 'Chemical Distance', filtered_df['Chemical Distance Similarity'])
                if show_active_only == 'Active':
                    filtered_df = filtered_df[filtered_df['Biological Activity'] == 'Active']
                elif show_active_only == 'Inactive':
                    filtered_df = filtered_df[filtered_df['Biological Activity'] == 'Inactive']
                if patho == "Coronaviruses":
                    filtered_df = filtered_df[(filtered_df['Pathogen'] == 'SARS_CoV') | (filtered_df['Pathogen'] == 'SARS_CoV-2')]
                # if patho == "SARS_CoV-2":
                #     filtered_df = filtered_df[filtered_df['Pathogen'] == 'SARS_CoV-2']
                if patho == "Leishmaniases":
                    filtered_df = filtered_df[filtered_df['Pathogen'] == 'Leishmania']
                filtered_df.drop(columns='Chemical Distance Similarity', inplace=True)
                filtered_df = filtered_df.loc[:, ~filtered_df.columns.str.contains('^Unnamed')]
                styled_filtered_df = filtered_df.style.applymap(highlight_active, subset=['Biological Activity'])
                styled_filtered_df = filtered_df.style.format({"Molecular Weight": "{:.2f}".format})
                if filtered_df.empty:
                    st.warning("No results found for the given SMILES.")
                else:
                    st.dataframe(styled_filtered_df ,
                                        column_config={
                                            "Chemical Distance": st.column_config.ProgressColumn(
                                                "Chemical Distance Similarity",
                                                help="Chemical Distance Similarity",
                                                format="%.3f",
                                                min_value=0,
                                                max_value=1,
                                            ),
                                             "refs": st.column_config.LinkColumn(),
                                             "Molecular Weight": st.column_config.NumberColumn(format="%.1f")
                                        }, hide_index=True)
                    @st.cache_data            
                    def convert_df(df):
                        return df.to_csv().encode('utf-8')

                    csv = convert_df(filtered_df)

                    st.download_button(
                        label="Download results as CSV",
                        data=csv,
                        file_name='results.csv',
                        mime='text/csv',)
                st.download_button(
                    label="Download All Data",
                    data=df_data.to_csv().encode('utf-8'),
                    file_name='all_data.csv',
                    mime='text/csv')
            except Exception as e:
                print(e)
                st.error(e)
                st.error('Invalid Smile')

    with tab2:
        pubchem_id = st.text_input(label='PUBCHEM ID', placeholder='161916')
        col1, col2, col3, col4= st.columns([0.4, 0.2, 0.2, 0.2])
        with col1:
            N = st.slider("Choose the number of closest molecules to display", 1,100,10, key=2)
        with col2:
            show_active_only = st.selectbox("Show Biological Activity", ['Active', 'Inactive', 'All'], key=9)
        with col3:
            distance = 'Sørensen–Dice'
            distance = st.selectbox('Distance Measure',('Sørensen–Dice', 'Tanimoto', 'Cosine', 'Tversky', 'Average'), key = 45)
        with col4:
            patho = "All"
            patho = st.selectbox('Pathogens',('Coronaviruses', 'Leishmaniases', 'All'), key = 4287)
        if not pubchem_id:
            pass
        else:
            try:
                smile_pub = pubchem_id_to_smiles(pubchem_id)
                with st.spinner("Please wait"):
                    try:
                        name = generate_name(smile_pub)
                        st.caption(name)
                    except:
                        print("cant generate name")
                df_smiles = df_data['SMILES'].tolist()
                similarities = calculate_similarity(smile_pub , df_smiles, np_data, distance_metric = distance)
                df_data['Chemical Distance Similarity'] = similarities
                df_data.sort_values(by='Chemical Distance Similarity', inplace=True, ascending=False)
                filtered_df = df_data.head(N)
                filtered_df.insert(0, 'Chemical Distance', filtered_df['Chemical Distance Similarity'])
                if show_active_only == 'Active':
                    filtered_df = filtered_df[filtered_df['Biological Activity'] == 'Active']
                elif show_active_only == 'Inactive':
                    filtered_df = filtered_df[filtered_df['Biological Activity'] == 'Inactive']
                if patho == "Coronaviruses":
                    filtered_df = filtered_df[(filtered_df['Pathogen'] == 'SARS_CoV') | (filtered_df['Pathogen'] == 'SARS_CoV-2')]
                # if patho == "SARS_CoV-2":
                #     filtered_df = filtered_df[filtered_df['Pathogen'] == 'SARS_CoV-2']
                if patho == "Leishmaniases":
                    filtered_df = filtered_df[filtered_df['Pathogen'] == 'Leishmania']
                filtered_df.drop(columns='Chemical Distance Similarity', inplace=True)
                filtered_df = filtered_df.loc[:, ~filtered_df.columns.str.contains('^Unnamed')]
                styled_filtered_df = filtered_df.style.applymap(highlight_active, subset=['Biological Activity'])
                styled_filtered_df = filtered_df.style.format({"Molecular Weight": "{:.2f}".format})

                if filtered_df.empty:
                    st.warning("No results found for the given SMILES.")
                else:
                    st.dataframe(styled_filtered_df ,
                                        column_config={
                                            "Chemical Distance": st.column_config.ProgressColumn(
                                                "Chemical Distance Similarity",
                                                help="Chemical Distance Similarity",
                                                format="%.3f",
                                                min_value=0,
                                                max_value=1,
                                            ),
                                             "refs": st.column_config.LinkColumn(),
                                             "Molecular Weight": st.column_config.NumberColumn(step=0.1)

                                        }, hide_index=True)
                    @st.cache_data            
                    def convert_df(df):
                        return df.to_csv().encode('utf-8')

                    csv = convert_df(filtered_df)

                    st.download_button(
                        label="Download results as CSV",
                        data=csv,
                        file_name='results.csv',
                        mime='text/csv',)
                st.download_button(
                    label="Download All Data",
                    data=df_data.to_csv().encode('utf-8'),
                    file_name='all_data.csv',
                    mime='text/csv')

            except Exception as e:
                print(e)
                st.error(e)
