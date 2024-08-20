import pandas as pd
import numpy as np
from typing import List, Dict

def assay_selection(df: pd.DataFrame, assay_title_colname:str='Assay Description') -> pd.DataFrame:
    """
    Filter bioassays with title containing one of 
    “Streptococcus pneumoniae”, "pneumococcus", or "pneumococci" 
    and one of “resistant”, "EryRc", or "PenR"
    and is not assessing other pneumoniae-causing pathogen klebsiella pneumoniae
    """
    eryRc = df[assay_title_colname].str.contains('EryRc')
    penR = df[assay_title_colname].str.contains('PenR')
    resistant = df[assay_title_colname].str.contains('resistant')
    klebsiella = df[assay_title_colname].str.contains('Klebsiella')

    cocci = df[assay_title_colname].str.lower().str.contains('pneumococci')
    strep = df[assay_title_colname].str.lower().str.contains('streptococcus pneumoniae') | df[assay_title_colname].str.lower().str.contains('s\. pneumoniae')
    df = df[(eryRc|penR|resistant) & (cocci|strep) & (~klebsiella)]
    return df 

def assay_type_filter(df: pd.DataFrame, assay_type_colname:str='Standard Type') -> pd.DataFrame:
    """
    only include assay type MIC, MIC90, MIC50, or Activity
    """
    type_filter = df[assay_type_colname].isin(['MIC', 'MIC90', 'MIC50']) | (df[assay_type_colname]=='Activity')
    df = df[type_filter]
    return df 

def standardize_MIC_units(df: pd.DataFrame, 
                 assay_unit_colname:str = 'Standard Units', 
                 assay_value_colname:str = 'Standard Value', 
                 molweight_colname:str = 'Molecular Weight') -> pd.DataFrame:
    """
    Convert other common MIC units to uM
    """
    df['adjusted_value_uM'] = np.nan
    mgml = df[assay_unit_colname].isin(['ug.mL-1','ug ml-1', 'mg/L'])
    nM = df[assay_unit_colname]=='nM'
    uM = df[assay_unit_colname]=='uM'
    df.loc[mgml, 'adjusted_value_uM'] = df[mgml][assay_value_colname] * 1000 / df[mgml][molweight_colname]
    df.loc[nM, 'adjusted_value_uM'] = df[nM][assay_value_colname] / 1000.00
    df.loc[uM, 'adjusted_value_uM'] = df[uM][assay_value_colname]
    return df

def MIC_to_classification_label(df:pd.DataFrame, 
                    MIC_threshold:int = 10, 
                    MIC50_threshold:int = 5,
                    assay_type_colname:str='Standard Type',
                    active_label_colname:str='Comment',
                    SMILES_colname:str='Smiles',
                    adjusted_MIC_colname:str='adjusted_value_uM'):
    """

    """
    activity_pos = (df[assay_type_colname]=='Activity') & (df[active_label_colname] == 'Active')
    mic_pos = (df[adjusted_MIC_colname] <= MIC50_threshold) | ((df[adjusted_MIC_colname] <= MIC_threshold ) & (df[assay_type_colname].isin(['MIC90','MIC'])))
    pos_dataset = df[activity_pos | mic_pos].drop_duplicates(SMILES_colname)

    activity_neg = (df[active_label_colname] == 'Not Active')
    mic_neg = (df[adjusted_MIC_colname] > MIC_threshold)
    neg_dataset = df[activity_neg | mic_neg].drop_duplicates(SMILES_colname)

    common = pos_dataset[[SMILES_colname]].merge(neg_dataset[[SMILES_colname]], on=SMILES_colname)
    common['key'] = 1
    pos_dataset = pos_dataset.merge(common, on=SMILES_colname, how='left')
    pos_dataset = pos_dataset[pos_dataset['key'].isna()]
    pos_dataset['final_activity_label'] = 1
    neg_dataset = neg_dataset.merge(common, on=SMILES_colname, how='left')
    neg_dataset = neg_dataset[neg_dataset['key'].isna()]
    neg_dataset['final_activity_label'] = 0
    final_dataset = pd.concat([pos_dataset, neg_dataset])
    return final_dataset

def final_reformat(df:pd.DataFrame, output_cols:List, rename_cols:Dict, data_source='ChEMBL') -> pd.DataFrame:
    df = df[output_cols].rename(columns=rename_cols)
    df['source'] = data_source
    return df

