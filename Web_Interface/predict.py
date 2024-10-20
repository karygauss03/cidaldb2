import traceback
import streamlit as st
from stmol import showmol
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
import pubchempy
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from scipy.spatial import distance
import pubchempy as pcp
import pickle
import os
import pickle
import torch
from streamlit_extras.add_vertical_space import add_vertical_space
import deepchem as dc
from deepchem.data import NumpyDataset
import dill
from transformers import RobertaTokenizerFast


def gcn_predictor(smiles_string, model):  
    featurizer = dc.feat.MolGraphConvFeaturizer()
    features = featurizer.featurize([smiles_string])
    dataset = NumpyDataset(X=features, y=None, ids=None)
    predictions = np.argmax(model.predict(dataset), axis=1)
    return predictions[0]

def chemebrta_predictor(smiles_string, model):
    tokenizer = RobertaTokenizerFast.from_pretrained('seyonec/SMILES_tokenized_PubChem_shard00_160k')
    
    inputs = tokenizer(smiles_string, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.item()

def load_pickle_files_from_folder(folder_path, name_condition=None):
    file_names = []
    
    for filename in os.listdir(folder_path):
        if name_condition is None or name_condition(filename):
            file_name_without_extension = os.path.splitext(filename)[0]
            file_names.append(file_name_without_extension)
    
    return file_names

folder_path = "./Web_Interface/models"

loaded_models_filenames = load_pickle_files_from_folder(folder_path, name_condition=lambda x: x.endswith('.pkl'))

def predict_with_model(smile, model_path):
    #st.text(type(model_path))
    #st.write(model_path)
    
    if model_path == "./Web_Interface/models/Coronavirus_GCN.pkl":
        #st.text("GCN")
        with open('./Web_Interface/models/Coronavirus_GCN.pkl', 'rb') as file:
            gcn_model = dill.load(file)
        # st.text(gcn_model.keys())
        y = gcn_predictor(smile, gcn_model)
        return y
    elif model_path == './Web_Interface/models/Covid_chemberta_model.pkl':
        with open(model_path, 'rb') as file:
            chemberta_model = dill.load(file)
        return chemebrta_predictor(smile, chemberta_model)
    elif model_path == './Web_Interface/models/Leishmania_chemberta_model.pkl':
        with open(model_path, 'rb') as file:
            chemebrta_model = dill.load(file)
        return chemebrta_predictor(smile,chemebrta_model)
    else:
        molecule = Chem.MolFromSmiles(smile)
        x = np.array(AllChem.RDKFingerprint(molecule, fpSize=2048))
        # Load the pickled model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Make the prediction using the loaded model
        y = model.predict([x])
    return y 

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


def predict():

    """
    ### Search By Smiles
    """

    st.title("Predict")
    st.write("Please select the predictive model corresponding to the pathogen you are interested in:")


    data = np.load("./Web_Interface/data/data.npy")



    tab1, tab2 = st.tabs(["Molecule SMILE",'PUBCHEM ID'])
    with tab1:
            smile = st.text_input(label = 'Molecule SMILE', placeholder = 'COC1=C(C=C(C=C1)F)C(=O)C2CCCN(C2)CC3=CC4=C(C=C3)OCCO4')
            option = st.selectbox(
                'Select Model',
                loaded_models_filenames, key = 42)
            if not smile:
                pass 
            else:
                try:
                    col1, col2 = st.columns([0.7, 0.3])
                    with col1:
                        name = generate_name(smile)
                        st.caption(name)
                        render = makeblock(smile)
                        render_mol(render)
                        progress_text = "Operation in progress. Please wait."
                    with col2:
                        with st.spinner(progress_text):
                            if predict_with_model(smile, f"./Web_Interface/models/{option}.pkl") == 1:
                                add_vertical_space(4)
                                st.success('Active', icon="✅")
                            elif predict_with_model(smile, f"./Web_Interface/models/{option}.pkl") == 0:
                                add_vertical_space(4)
                                st.error('Inactive', icon="❌")
                except Exception as e:
                    # st.error(e)
                    st.error(traceback.format_exc())
                    st.error("Invalid Smile")  # You might want to adjust this message based on the context of the error




    with tab2:
        pub = st.text_input(label = 'PUBCHEM ID', placeholder = '161916')
        option = st.selectbox('Select Model',loaded_models_filenames, key = 43)
        if not pub:
            pass 
        else:
            try:
                cola, colb = st.columns([0.7, 0.3])
                with cola:
                    smile = pubchem_id_to_smiles(pub)
                    name = generate_name(smile)
                    st.caption(name)
                    render = makeblock(smile)
                    render_mol(render)
                    progress_text = "Operation in progress. Please wait."
                with colb:
                    with st.spinner(progress_text):
                        if predict_with_model(smile, f"./Web_Interface/models/{option}.pkl") == 1:
                            st.success('Active', icon="✅")
                        elif predict_with_model(smile, f"./Web_Interface/models/{option}.pkl") == 0:
                            st.error('Inactive', icon="❌")
            except Exception as e:
                print(e)
                st.error('Invalid PubchemID')
