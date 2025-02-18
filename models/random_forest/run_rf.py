from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score, auc, matthews_corrcoef, precision_score, recall_score

# read in data
df = pd.read_csv(r'../../processed_datasets/10uM_FP_clustered__resistant_pneumococcus_augmented_dataset_v7.csv')
train_idx = df[df.cluster_id!=-1].index.values 
test_idx = df[df.cluster_id==-1].index.values

# generate 2048-bit morgan fingerprint of radius 3
train_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 3, 2048, useChirality=True) for smiles in df.loc[train_idx]['Smiles'].values]
test_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 3, 2048, useChirality=True) for smiles in df.loc[test_idx]['Smiles'].values]

# hyperparameter tuning
params = {'max_depth':[None, 16, 256],
          'min_samples_leaf':[1, 2, 4],
          'criterion':['gini','entropy'],
          'class_weight':['balanced',None]}

rfc = RandomForestClassifier(n_estimators=200, random_state=3)
rfc_cv = GridSearchCV(rfc, params, cv =5, n_jobs=5, scoring='average_precision')
rfc_cv.fit(train_fps, df.loc[train_idx]['final_activity_label'])
joblib.dump(rfc_cv.best_estimator_, r'../../model_garden/tuned_rfc_morgan_seed3.joblib')

rfc_cv_resultsDF = pd.DataFrame(rfc_cv.cv_results_).sort_values('mean_test_score', ascending=False)
rfc_cv_resultsDF.to_csv(r'../../processed_datasets/rfc_morgan_5StratifiedFold_CV_results.csv', index=False)
print(rfc_cv.best_estimator_)

pred = [x[1] for x in rfc_cv.predict_proba(test_fps)]
p_precision, p_recall, thresholds = precision_recall_curve(df['final_activity_label'].loc[test_idx], pred)
print('\tMCC @0.5: ', matthews_corrcoef(df['final_activity_label'].loc[test_idx], [x >0.5 for x in pred]))
print('\tprecision: ', precision_score(df['final_activity_label'].loc[test_idx], [x >0.5 for x in pred]))
print('\trecall: ', recall_score(df['final_activity_label'].loc[test_idx], [x >0.5 for x in pred]))
print('\tAUPRC: ', auc(p_recall, p_precision))
print('\tROC-AUC score: ', roc_auc_score(df['final_activity_label'].loc[test_idx], pred))