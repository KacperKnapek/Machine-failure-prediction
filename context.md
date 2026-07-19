PROJECT=AI4I2020 binary machine failure prediction.
STATUS=baseline reproducible.
LANG=python sklearn pandas.

DATA=10000 rows; target=Machine failure; positives=330 after correcting 9 inconsistent labels.
SPLIT=train_test_split(test_size=0.2,stratify=y,random_state=42).

FEATURES=Type_L,Type_M,ProcessTemp,TempDiff,RPM,Torque,Power,ToolWear.
ENGINEERING=Power=rpm*torque*2*pi/60; TempDiff=Process-Air.
DROP=UDI,ProductID,TWF,HDF,PWF,OSF,RNF,record_index.
NO_LEAKAGE=never use failure flags.

PIPELINE=cleaning.py->data_preparation.py->EDA->modeling.
OUTPUT=X_train_raw,X_test_raw,X_train_scaled,X_test_scaled,y_train,y_test.
INDEX=record_index only for joins.

SCALER=StandardScaler fit(train) transform(train,test) numeric only.
MODELS=Dummy,LogReg(balanced),RandomForest(200,balanced),GradientBoosting.
THRESHOLD=0.5.
PRIMARY_METRICS=PR-AUC,F1,Recall,Precision,ROC-AUC,FN,FP; accuracy secondary.

BASELINE=RF recall≈0.83 precision≈0.85 ROC≈0.991; GB precision≈0.88 recall≈0.79 ROC≈0.994; LR high recall low precision.

NEXT=cv,tuning,threshold optimization,error analysis,feature engineering,calibration,SHAP,final evaluation.

RULES=keep reproducible; no leakage; keep record_index; compare models on same split; document every experiment.

EXPERIMENT_NOTEBOOK=notebooks/05_feature_experiment.ipynb.
EXPERIMENT=compare baseline vs baseline+OSF using 5-fold StratifiedKFold on training data.
OSF_FEATURE=Tool wear [min] * Torque [Nm].
CV_SCALING=fit StandardScaler inside each fold on numeric train columns only.
OOF=out-of-fold probabilities are used for threshold analysis and confusion matrices.
THRESHOLDS=test 0.20,0.30,0.40,0.50,0.60,0.70,0.80; default decision remains >=0.5.
COST_ANALYSIS=compare FP/FN costs with example scenarios; business costs not yet known.

BEST_CURRENT=GradientBoosting + OSF at threshold 0.5.
X_TEST_RESULT=precision 0.9825; recall 0.8485; F1 0.9106; ROC-AUC 0.9953; PR-AUC 0.9357; FP 1; FN 10.
X_TEST_CM=[[1933,1],[10,56]].
TEST_CAVEAT=X_test was used during exploration and OSF design; treat this as a development check, not untouched external validation.
FOLD5_NOTE=GB+OSF fold 5 had precision 1.0 and recall 0.6981 with [[1547,0],[16,37]]; similar recall degradation in RF suggests a harder fold.

HELPERS=python/features.py(add_osf_criterion);python/evaluation.py(prepare_fold_data,get_metrics,cross_validate,summarize_fold_metrics,evaluate_thresholds);python/visualization.py(plot_confusion_matrix_grid,plot_threshold_analysis).
HELPERS_NOTE=cross_validate returns fold metrics + OOF probabilities from one training pass per fold; verified bit-identical to notebook 05 loops and to documented GB+OSF fold5 result.

SENSITIVITY=python/sensitivity_analysis.py compares corrected(330) vs uncorrected(339) labels; 5-fold CV RF+GB on baseline+OSF; conclusion=differences within fold std, model ranking unchanged, correction not driving results; output=results/sensitivity_target_correction.csv.
TIES_CONTROL=evaluation.get_threshold_ties + ties column in evaluate_thresholds; notebook 04 prints proba==threshold records (RF: 5653).
EXPORTS=false_positives_baseline.csv includes raw feature columns again; report probabilities rounded to 4 decimals, computations keep full precision.

FEATURE_IMPORTANCE=notebook 05; GB+OSF; impurity(final model) + permutation(per CV fold, PR-AUC, n_repeats=10); top group=Power,TempDiff,RPM,OSF; OSF confirmed real signal (impurity 0.2437, permutation 0.2267±0.0475); raw Torque~0 and ProcessTemp~0 (removal candidate); correlated features caveat=read ranking as a group.

REDUCED_SET=notebook 05; GB+OSF minus Torque minus ProcessTemp (7 features); X_test identical to 9-feature model (precision 0.9825, recall 0.8485, FP 1, FN 10; PR-AUC 0.9368 vs 0.9357); CV slightly lower (F1 0.8900 vs 0.8954) but within fold std; reduced set is a candidate for the final model.
TEMPERATURE_NOTE=absolute ProcessTemp carries no signal (class means 310.27 vs 310.00, corr with target 0.033); failure signal is the gradient TempDiff (9.38 vs 10.02, corr -0.114, HDF band 7.6-8.6); ProcessTemp=ambient+~10K so absolute level tracks ambient drift; conclusion is dataset-specific redundancy given TempDiff, not general unimportance of temperature.

FINAL_MODEL=python/final_model.py; GB(random_state=42); 7 features (no raw Torque, no ProcessTemp); OOF cost thresholds: 1x->0.60, 5-10x->0.30, 20x->0.15, 30-50x->0.05; recommendation 0.30 for 5-10x (X_test: precision 0.892, recall 0.879, FP 7, FN 8), 0.5 for precision-first (0.983/0.848, FP 1, FN 10); artifact=models/final_model.joblib (model+scaler+features+per-ratio thresholds).

NEXT=obtain real FP/FN business costs and fix the threshold; validate on future/external data; probability calibration; drift monitoring.
