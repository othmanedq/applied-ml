# Term deposit subscription model

A Portuguese bank calls its clients to sell a term deposit. This project predicts, just before a
call, whether the client will subscribe, and serves the model through an API.

- **Data:** [Bank Marketing](https://www.openml.org/d/1461) (OpenML 1461), from the telemarketing
  campaigns of a Portuguese bank (Moro, Cortez & Rita, 2014).
- **Final model:** random forest, with a decision threshold chosen on the validation set.
- **Sensitive attribute:** age group (18-29, 30-59, 60+).

This README describes the method and the reasons behind each choice. It contains no numbers: every
number is computed by the code and saved in `outputs/`, and `notebooks/design_choices.ipynb` shows
the evidence behind each choice. References are listed at the end; "course" refers to the sessions
of the Applied ML course (Sebag, 2026).

## Where to find the results

| Question | File |
|---|---|
| Test scores of every model, with bootstrap intervals | `outputs/tables/evaluation_test.csv`, `outputs/tables/evaluation_test_intervals.csv` |
| Final model and decision threshold | `outputs/evaluation_selection.json` |
| Model screening before the tuning | `outputs/tables/screening_models.csv` |
| Every tuning trial (run log) | `outputs/tables/tuning_run_log.csv`, `outputs/figures/tuning_history.png` |
| Class weights experiment | `outputs/tables/imbalance_class_weights.csv` |
| Feature and time-proxy ablations | `outputs/tables/ablation_features.csv`, `outputs/tables/ablation_time_proxies.csv` |
| Time split check | `outputs/tables/time_split_check.csv`, `outputs/tables/time_split_intervals.csv` |
| Calibration | `outputs/calibration_summary.json`, `outputs/figures/calibration_curve.png` |
| SHAP explanations | `outputs/tables/shap_importance.csv`, `outputs/tables/shap_cases.csv`, `outputs/figures/shap_*.png` |
| Fairness before and after mitigation | `outputs/tables/fairness_summary.csv`, `outputs/tables/fairness_groups.csv` |
| Exploratory data analysis | `outputs/tables/eda_*.csv`, `outputs/figures/eda_*.png` |

## Setup

Python 3.13 is required.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset is downloaded from OpenML on the first run and then cached by scikit-learn.

## Usage

```bash
python main.py            # all steps; the tuning step takes the longest
python -m pytest          # tests
```

Each step can also run alone. For example, this command reuses the saved Optuna results
(`outputs/tuning_best_params.json`) and skips the tuning:

```bash
python main.py --steps train evaluate time_split calibration explain fairness
```

The steps are, in order: `eda`, `imbalance`, `screening`, `tune`, `ablation`, `train`, `evaluate`,
`time_split`, `calibration`, `explain`, `fairness`. All settings (seed, split sizes, number of folds,
trials and bootstrap resamples, age groups, SHAP sample sizes) are in `config.yaml`. The trained models are saved in `models/`,
which is not versioned, so run `train` after a fresh clone.

### Prediction API

```bash
uvicorn src.api:app
```

The API documentation is at http://127.0.0.1:8000/docs, and `GET /health` returns the name of the
model and its threshold. Example request, for a made-up client:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '[{
  "age": 35, "job": "management", "marital": "married", "education": "tertiary",
  "default": "no", "balance": 1500, "housing": "no", "loan": "no", "contact": "cellular",
  "day": 15, "month": "may", "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "unknown"
}]'
```

For each client, the answer gives the predicted probability of subscription and `call`, which is
true when this probability reaches the decision threshold. The API rejects unknown categories,
impossible values and empty lists with an HTTP 422 error.

### Notebook

`notebooks/design_choices.ipynb` justifies the choices of the pipeline. Each section asks one
question, shows the evidence computed on the training and validation sets, and gives the decision.
Part 1 (the data) only needs the dataset. Part 2 (modelling choices) loads the models trained by
`python main.py` and reads the Optuna run log, so run `python main.py` first. Open it with
`jupyter lab` or VS Code.

## Project structure

```
├── main.py              # runs the steps in order
├── config.yaml          # all settings
├── requirements.txt     # exact versions of all packages
├── src/
│   ├── data.py          # loading, cleaning, train/validation/test split, age groups
│   ├── eda.py           # exploratory data analysis
│   ├── features.py      # feature engineering and preprocessing
│   ├── models.py        # the scikit-learn Pipeline of each model and its search space
│   ├── imbalance.py     # class weights experiment
│   ├── tuning.py        # model screening, Optuna search and run log
│   ├── ablation.py      # feature and time-proxy ablations
│   ├── evaluation.py    # metrics, threshold, bootstrap intervals, figures, time split
│   ├── calibration.py   # reliability curve, Brier score, ECE
│   ├── explain.py       # SHAP values
│   ├── fairness.py      # fairness metrics and group thresholds
│   ├── api.py           # FastAPI application
│   └── utils.py         # paths, config loading, saving and loading files
├── tests/               # pytest tests
├── notebooks/           # data exploration and design choices
└── outputs/             # tables (CSV), figures (PNG) and key numbers (JSON)
```

The output files start with their topic (`eda_`, `imbalance_`, `screening_`, `tuning_`,
`ablation_`, `evaluation_`, `time_split_`, `calibration_`, `shap_`, `fairness_`).

## Method

The section of `notebooks/design_choices.ipynb` that holds the evidence is given in brackets.

### Problem framing

One row is one call. The model must decide whom to call, so it may only use what is known before
the call (course, session 2).

- `duration`, the length of the call, is removed when the data is loaded. It is only known after
  the call, so using it would be target leakage (Kaufman et al., 2012). The authors of the data give
  the same warning: duration "should be discarded if the intention is to have a realistic predictive
  model" (file `bank-additional-names.txt` of the UCI dataset; Moro, Rita & Cortez, 2014). [1.1]
- `campaign`, `day` and `month` are kept: they describe the call that is about to be made.

### Data preparation

- There are no missing values (NaN), but some columns use the category `unknown`. It is kept as its
  own category and one-hot encoded, because the fact that a value is missing can carry information.
  [1.3]
- Large values are kept: they are possible values, not errors. The usual IQR rule flags every
  client contacted before, so it is not used to remove rows. A tree only compares each feature with
  a threshold, so monotone transformations of the inputs do not change it (course, session 4). [1.4]
- Correlations use the Spearman coefficient, because several features are very skewed. Correlated features are kept: they do not hurt the trees, and the L2 penalty of the
  logistic regression shares the weight between them (course, session 3). [1.5]

### Split

The data is split at random into train, validation and test sets, stratified on the target, with a
fixed seed (`config.yaml`; course, session 2).

- **Train:** EDA, cross-validation, tuning, and fitting of the final models.
- **Validation:** choice of the model, of the decision threshold, and of the fairness thresholds.
- **Test:** only for the final scores. No choice is made from the test set, because a test set
  that guides choices becomes optimistic (course, session 2).

The rows are in time order, and the subscription rate changes a lot over time. This is a dataset
shift (Moreno-Torres et al., 2012). A random split mixes the periods, so its scores are same-period
scores, and they are reported as such. For data that changes over time, the standard is to evaluate
on the newest rows (Bergmeir & Benítez, 2012), so a time split check measures how optimistic the
scores are. [1.9, 2.10]

### Features and preprocessing

Everything is inside one scikit-learn `Pipeline`: feature engineering, then a `ColumnTransformer`,
then the model. The preprocessing is therefore fitted again inside every cross-validation fold,
which prevents leakage (course, session 6). The API uses the same
fitted pipeline, so new clients get exactly the same preprocessing.

- Categories are one-hot encoded (course, session 6). A category never seen in training is encoded
  as zeros.
- Numeric features are standardized only for the models that need it: the logistic regression,
  because the L2 penalty depends on the scale of the features (course, session 3), and the
  k-nearest neighbours of the model screening. Trees do not need it.
- Two features are added: `has_loan` (housing loan or personal loan) and `contacted_before`
  (contacted in a previous campaign). Both follow from the EDA. The total number of contacts
  (`campaign + previous`) was rejected, because its two parts go in opposite directions. [1.6]
- An ablation shows that the two features do not improve any model: the models already find the
  same information in the original columns. They are kept because they cost nothing and each one
  states a business fact in one column. [2.4]
- Two more candidates were tested with a rule fixed in advance: a feature is added only if it
  improves a model by more than the standard deviation across folds. Age groups (for the U-shaped
  effect of age) and a signed log of the skewed features are both rejected. [2.4]

### Metric and class imbalance

Subscriptions are rare, so accuracy is misleading. The main metric is **average precision** (AP),
which summarises the precision-recall curve. On imbalanced data, the precision-recall curve is more
informative than the ROC curve, and a model with no skill gets an AP equal to the share of
positives (Saito & Rehmsmeier, 2015; course, session 9). We also report ROC-AUC, the Brier score
(the mean squared error of the probabilities), and precision, recall, F1 and balanced accuracy at
the chosen threshold.

- **No class weights.** In cross-validation, `class_weight="balanced"` does not improve AP but makes
  the Brier score worse [2.1]. This agrees with van den Goorbergh et al. (2022): imbalance
  corrections such as oversampling, which class weights imitate, damage the probabilities without
  improving discrimination, and moving the threshold gives the same balance between sensitivity
  and specificity.
- **Decision threshold** chosen on the validation set: the one with the best balanced accuracy,
  which is the maximum of Youden's index. This threshold minimises the expected
  cost under one assumption: missing one subscriber costs as much as N useless calls, where N is
  the number of non-subscribers per subscriber (Elkan, 2001). The bank should check this cost
  ratio. [2.7]

### Models and tuning

The project started with the three model families of the course: logistic regression, random
forest and LightGBM. To check this choice, eight models from six families were then compared with
their default settings in cross-validation. The rule for changing the choice was fixed before
running this comparison: each family is represented by its best model, one of the three models is
replaced only if another model of its family is better by more than the standard deviation across
folds, and the three best families are the ones that are tuned. The comparison confirms the
choice. [2.2]

| Family | Models compared | Reference |
|---|---|---|
| Linear | Logistic regression | course, session 2 |
| Single tree | Decision tree | course, session 4 |
| Bagging | Random forest, extra trees | Breiman (2001); course, session 5 |
| Boosting | LightGBM, HistGradientBoosting | Ke et al. (2017); scikit-learn documentation; course, session 5 |
| Naive Bayes | Gaussian naive Bayes | – |
| Nearest neighbours | k-nearest neighbours | – |

- An SVM was not tested: it gives no probabilities without an extra calibration step, and its fit
  time grows at least quadratically with the number of rows (scikit-learn, `SVC` documentation).
- Neural networks were not tested: on tabular data of this size, tree ensembles usually do at
  least as well (Grinsztajn et al., 2022).
- The three best families are boosting, bagging and linear. The project keeps the baseline (which
  predicts the subscription rate for everyone), the logistic regression, the random forest and
  LightGBM: from the simplest model to the most flexible, so the comparison shows what each step
  adds.

Each model is tuned with Optuna (Akiba et al., 2019) and its TPE sampler (course, session 6). Each
trial scores the whole pipeline with stratified cross-validation on the
training set; the objective is the mean AP. The number of trees of the random forest is fixed and
not tuned, because more trees do not hurt (Probst & Boulesteix, 2018). [2.3]

The best cross-validation score is the best of many trials, so it is optimistic (Cawley & Talbot,
2010; course, session 6). This is why the models are compared again on the validation set, and the
final scores come from the test set. Nested cross-validation (course, session 6) would give a
more stable estimate; a single validation set and a single test set are cheaper, but the scores
depend on one split.

**What the trees add.** One diagnostic feature, the month and day of the call as one category,
brings the logistic regression to the level of the tree models, and the trees lose a clear part of
their lead without `day`. Most of what the trees add is therefore this one interaction. The dates
with a high subscription rate have few calls and are mostly in the newest part of the data, so the
date is one more way to recognise the recent period. The feature is not added to the pipeline: it
brings nothing to the tree models, so it would not change the final model. [2.4, 2.5]

**Model choice.** The rule, fixed before the experiments, keeps the model with the best validation
AP. The random forest and LightGBM are tied: their difference is smaller than the variation
between folds, and the bootstrap interval of their gap on the test set contains zero (Efron &
Tibshirani, 1993). The rule picks the random forest. Changing the rule
after seeing the results would bias the estimate (Cawley & Talbot, 2010). [2.6]

**Time proxies.** `contact`, `month` and `day` partly tell the model when the call happened.
Removing each of them was tested with a rule fixed in advance for `contact` and `month`, and
applied as it is to `day`; the three columns are kept. `contact` remains a known weakness: without
it, the models are a little worse on the same period but a little better on newer calls. [2.5]

### Time split check

Every model is trained again on the oldest rows of train + validation and scored on the newest
ones. AP depends on the share of positives (Saito & Rehmsmeier, 2015), so we compare ROC-AUC, which
does not. The check has limits: it is a single split, the change of the subscription rate and the
change of the features are mixed, and its rows were already used by our choices. [2.10]

Every model loses, and the lead of the trees is gone: on the newest rows the logistic regression
goes from last to first and the random forest ends last. The bootstrap interval of the gap between
these two does not contain zero (Efron & Tibshirani, 1993). `contact` explains a part of it; the
date of the call does not; the rest of the cause is not identified. [2.5, 2.10]

The final model is not changed: the protocol asks which model is the best on the same period, and
the rule was fixed before the results. But this check shows what the protocol cannot do: it cannot
say which model the bank should use for a new campaign. For that, the three models should be
tuned, compared and calibrated again with a validation in time order.

### Calibration

The final model is checked with a reliability curve (scikit-learn, "Probability calibration"), the
Brier score and the expected calibration error (ECE) with equal-width bins (Guo et al., 2017). With equal-width bins, the ECE
is dominated by the bins that hold most clients, so the bins are also read one by one (Nixon et
al., 2019).

On the validation set, the predictions are slightly too high for the lowest probabilities and too
low in the middle of the range. This shape is known for averaging models such as random forests:
they have difficulty predicting probabilities near 0 and 1 (scikit-learn, "Probability
calibration"). On the test set, the highest probabilities are too high
instead; these bins hold few clients, so the top of the range is uncertain.

The decision not to recalibrate is taken on the validation set: the model is used to rank clients
and to apply a threshold. The probabilities should not be used as exact values, nor for a new
campaign. [2.8]

### Explainability

SHAP values (Lundberg & Lee, 2017; course, session 9) are computed with `TreeExplainer`
(Lundberg et al., 2020) on a sample of test clients (`config.yaml`). For the random forest they are
in probability: the base value plus the sum of the SHAP values gives the predicted probability.
Because SHAP values add up, the values of the one-hot columns of a feature are summed to give one
value per original feature.

The two most important features are `month` and `contact`. They must be read with care: without a
year column, they partly tell the model **when** the call happened, not who the client is. The next
features have a clear business sense (`outputs/figures/shap_beeswarm.png`): the result of the
previous campaign, having a loan, the age (young and older clients subscribe more) and the
balance. The local explanations show the most confident true positive, false positive, false
negative and true negative.

### Fairness

Age is a protected ground of non-discrimination in EU law (Charter of Fundamental Rights of the
European Union, article 21).

- **Definition: equal opportunity** (Hardt, Price & Srebro, 2016): among the clients who would
  subscribe, every age group should have the same chance to be called, so the true positive rate
  (recall) should be the same. Demographic parity, the same share of calls in every group (Barocas,
  Hardt & Narayanan, 2023), was rejected, because the real subscription rates are
  very different between age groups. [2.9]
- **A gap that is partly the drift:** the oldest group is concentrated in the newest part of the
  data, where the model gives high probabilities to every client. [1.9, 2.9]
- **Mitigation: post-processing**, with one threshold per age group, chosen on the validation set
  so that every group reaches the overall recall of the single threshold (Hardt, Price & Srebro,
  2016). The rule is deterministic: two identical clients always get the same decision. fairlearn's
  `ThresholdOptimizer` was rejected because its predictions are randomized (fairlearn
  documentation).
- **Uncertainty:** the 60+ group is small, so the rates of each group are reported with 95% Wilson
  intervals, which stay reliable for small groups (Brown, Cai & DasGupta, 2001).
- **Cost:** the mitigation makes more calls useless. The API serves the model with a single
  threshold, because the group thresholds would need the client's age at decision time, which is a
  legal and business choice for the bank.

## Limitations

- The scores come from a random split and are optimistic for a future campaign (see the time split
  check). A real deployment would need a time-based validation and regular recalibration.
- On newer calls, the final model ranks clients less well than the logistic regression. The
  `contact` column explains a part of it; the rest of the cause is not identified.
- The period of the call is the strongest signal in this data: it is behind the two most important
  features, a large part of the lead of the trees, and a part of the fairness gap.
- `contact`, `month` and `day` partly encode the period of the call, not the client.
- The 60+ group is small, so its rates have wide intervals.
- The model choice rests on one validation split and one seed, and the random forest and LightGBM
  are tied.
- The decision threshold rests on an assumed cost ratio between a missed subscriber and a useless
  call.
- The highest predicted probabilities are too high on the test set, in bins with few clients.
- The engineered features do not improve the models, and neither do the two other candidates
  that were tested.

## References

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation
  hyperparameter optimization framework. *Proceedings of KDD 2019*, 2623–2631.
- Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning: Limitations and
  Opportunities*. MIT Press. https://fairmlbook.org
- Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor
  evaluation. *Information Sciences*, 191, 192–213.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial proportion.
  *Statistical Science*, 16(2), 101–133.
- Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent
  selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079–2107.
- Charter of Fundamental Rights of the European Union (2012/C 326/02), article 21.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Elkan, C. (2001). The foundations of cost-sensitive learning. *Proceedings of IJCAI 2001*,
  973–978.
- fairlearn documentation. Postprocessing: `ThresholdOptimizer`.
  https://fairlearn.org/main/user_guide/mitigation/postprocessing.html
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform
  deep learning on typical tabular data? *Advances in Neural Information Processing Systems 35,
  Datasets and Benchmarks Track*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural
  networks. *Proceedings of ICML 2017*, PMLR 70, 1321–1330.
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning.
  *Advances in Neural Information Processing Systems 29*.
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining:
  Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4),
  article 15.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A
  highly efficient gradient boosting decision tree. *Advances in Neural Information Processing
  Systems 30*.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions.
  *Advances in Neural Information Processing Systems 30*.
- Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb,
  J., Bansal, N., & Lee, S.-I. (2020). From local explanations to global understanding with
  explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67.
- Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V., & Herrera, F. (2012). A
  unifying view on dataset shift in classification. *Pattern Recognition*, 45(1), 521–530.
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank
  telemarketing. *Decision Support Systems*, 62, 22–31.
- Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C5K306
- Nixon, J., Dusenberry, M. W., Zhang, L., Jerfel, G., & Tran, D. (2019). Measuring calibration in
  deep learning. *CVPR Workshops 2019*, 38–41.
- Probst, P., & Boulesteix, A.-L. (2018). To tune or not to tune the number of trees in random
  forest. *Journal of Machine Learning Research*, 18(181), 1–18.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC
  plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432.
- scikit-learn documentation. Probability calibration.
  https://scikit-learn.org/stable/modules/calibration.html
- scikit-learn documentation. `HistGradientBoostingClassifier` and `SVC`.
  https://scikit-learn.org/stable/api/index.html
- Sebag, I. (2026). *Applied ML: Foundations, Engineering & Evaluation*, sessions 2, 3, 4, 5, 6
  and 9. Albert School.
- van den Goorbergh, R., van Smeden, M., Timmerman, D., & Van Calster, B. (2022). The harm of class
  imbalance corrections for risk prediction models: illustration and simulation using logistic
  regression. *Journal of the American Medical Informatics Association*, 29(9), 1525–1534.
