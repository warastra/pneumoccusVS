import requests
from tqdm import tqdm 
from io import StringIO
import time 
import pandas as pd
from bs4 import BeautifulSoup
from typing import List

def read_pubchem(assay_id) -> pd.DataFrame:
    template_json_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{}/json"
    template_csv_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{}/csv"
    if not isinstance(assay_id, str):
        assay_id = str(assay_id)

    assay_json = requests.get(template_json_url.format(assay_id)).json()
    assay_title = [x.replace('Title: ','') for x in assay_json['PC_AssaySubmit']['assay']['descr']['description'] if 'Title' in x][0]
    assay_doi = [x for x in assay_json['PC_AssaySubmit']['assay']['descr']['comment'] if 'DOI' in x][0]
    assay_chembl_id = assay_json['PC_AssaySubmit']['assay']['descr']['aid_source']['db']['source_id']['str']

    assay_req = requests.get(template_csv_url.format(assay_id))
    assay_data = pd.read_csv(StringIO(assay_req.text))
    assay_data = assay_data[~assay_data['PUBCHEM_SID'].isna()]

    assay_data['Assay Description'] = assay_title
    assay_data['assay_doi'] = assay_doi
    assay_data['assay_id'] = assay_chembl_id
    return assay_data

def get_chembl_ids(df:pd.DataFrame, substanceID_colname:str='sid') -> List:
    extracted_chemblid = []
    for idx, sid in tqdm(enumerate(df[substanceID_colname].values)):
        # wait a while to get around http request limit
        if idx != 0 and idx % 500==0:
            time.sleep(30)
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sid/{sid}/XML'
        xmlfile = requests.get(url)
        soup = BeautifulSoup(xmlfile.text, 'html.parser')
        extracted_chemblid.append(soup.find_all('object-id_str')[0].text)
    
    return extracted_chemblid

def get_pubchem_bioactivity(bioassayDF, assay_id_colname:str='aid') -> pd.DataFrame:
    Pubdf = pd.DataFrame()
    for idx, assay_id in tqdm(enumerate(bioassayDF[assay_id_colname].unique())):
        # wait a while to get around http request limit
        if idx != 0 and idx % 500==0:
            time.sleep(30)
        assay_data = read_pubchem(assay_id)
        Pubdf = pd.concat([Pubdf, assay_data])
    return Pubdf