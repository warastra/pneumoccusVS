import pandas as pd
import numpy as np
from compileData_utils import *
from getPubChem_utils import get_chembl_ids, get_pubchem_bioactivity
from rdkit.Chem import Descriptors
from rdkit import Chem

MRSA_ASSAY_CUTOFF = 0.5
PubChem_COL_RENAME = {
    'PUBCHEM_ACTIVITY_OUTCOME':'Comment',
    'PUBCHEM_EXT_DATASOURCE_SMILES':'Smiles'
}
PubChem_DATA_DETAILS = {
        'data_source':'PubChem',
        'assay_title_colname':'Assay Description',
        'assay_type_colname':'Standard Type',
        'assay_unit_colname':'Standard Units', 
        'assay_value_colname':'Standard Value', 
        'molweight_colname':'Molecular Weight',
        'active_label_colname':'PUBCHEM_ACTIVITY_OUTCOME',
        'Smiles_colname':'PUBCHEM_EXT_DATASOURCE_SMILES',
        'Compound_ID_colname':'Molecule ChEMBL ID'
    }
ChEMBL_DATA_DETAILS = {
        'data_source':'ChEMBL',
        'assay_title_colname':'Assay Description',
        'assay_type_colname':'Standard Type',
        'assay_unit_colname':'Standard Units', 
        'assay_value_colname':'Standard Value', 
        'molweight_colname':'Molecular Weight',
        'active_label_colname':'Comment',
        'Smiles_colname':'Smiles',
        'adjusted_MIC_colname':'adjusted_value_uM',
        'Compound_ID_colname':'Molecule ChEMBL ID'
    }

if __name__ == '__main__':
    # compile and filter ChEMBL data
    chembl2 = pd.read_csv('raw_datasets\CheMBL_pneumococci_raw_activity.csv', delimiter=';')
    chembl = pd.read_csv('raw_datasets\CheMBL_S_pneumoniae_raw_activity.csv', delimiter=';')
    chembl = pd.concat([chembl, chembl2])
    # chembl = chembl.drop_duplicates(['Molecule ChEMBL ID', 'Standard Type'])

    chembl = assay_selection(chembl, assay_title_colname='Assay Description')
    chembl = assay_type_filter(chembl)
    chembl['source'] = 'ChEMBL'

    # compile and filter PubChem data
    pubchem_bioassays = pd.read_csv('raw_datasets/pubchem_targettaxid_1313_bioactivity.csv')
    pubchem = get_pubchem_bioactivity(pubchem_bioassays)
    pubchem = pubchem.rename(columns=PubChem_COL_RENAME)
    pubchem = pubchem[~pubchem['Smiles'].isna()]    # remove data with missing SMILES
    pubchem = assay_selection(pubchem, assay_title_colname='Assay Description')
    pubchem = assay_type_filter(pubchem)
    pubchem['Molecular Weight'] = [Descriptors.ExactMolWt(Chem.MolFromSmiles(x)) for x in pubchem['Smiles']]
    pubchem['Molecule ChEMBL ID'] = get_chembl_ids(pubchem, substanceID_colname='PUBCHEM_SID')
    pubchem['source'] = 'PubChem'
    
    # combine ChEMBL & PubChem, label data and deduplicate, prioritizing ChEMBL entry
    combined_dataset = pd.concat([chembl, pubchem]).reset_index(drop=True)
    combined_dataset = standardize_MIC_units(combined_dataset)
    combined_dataset = MIC_to_classification_label(combined_dataset, MIC_threshold=10, MIC50_threshold=5)
    combined_dataset = combined_dataset.rename(columns={"Molecule ChEMBL ID":"Compound ID"})

    # augment with negative instances from assay on methicillin-resistant S. aureus
    mrsa_xls = pd.ExcelFile("Wong_2024_explainableDL_structural_class_MOESM3_ESM_data.xlsx")
    mrsa_neg_dataset = mrsa_xls.parse('S. aureus growth inhibition')

    mrsa_neg_dataset = mrsa_neg_dataset[mrsa_neg_dataset['Mean_50uM']>MRSA_ASSAY_CUTOFF].drop_duplicates(subset=['Compound_ID']).reset_index(drop=True)
    mrsa_neg_dataset = mrsa_neg_dataset.rename(columns={'SMILES':'Smiles', 'Mean_50uM':'Standard Value', 'Compound_ID':'Compound ID'})
    mrsa_neg_dataset['Molecular_Weight'] = [Descriptors.ExactMolWt(Chem.MolFromSmiles(x)) for x in mrsa_neg_dataset['Smiles']]

    mrsa_neg_dataset['final_activity_label'] = 0
    mrsa_neg_dataset['Standard Type'] = 'Growth Inhibition'
    mrsa_neg_dataset['Standard Units'] = 'normalized %'
    mrsa_neg_dataset = mrsa_neg_dataset.drop(columns=['R1_50uM','R2_50uM'])
    mrsa_neg_dataset['source'] = 'MRSA_Wong2024'

    augmented_dataset = pd.concat([combined_dataset, mrsa_neg_dataset]).reset_index(drop=True)
    augmented_dataset = augmented_dataset.drop_duplicates(subset=['Smiles', 'final_activity_label'])
    
    # write augmented dataset to disk
    augmented_dataset.to_csv(r'processed_datasets\10uM_resistant_pneumococcus_augmented_dataset_v7.csv', index=False) #

    # print data summary
    print(augmented_dataset['final_activity_label'].value_counts())
    print(augmented_dataset.groupby(['final_activity_label', 'source','Standard Type'])['Smiles'].count())



    