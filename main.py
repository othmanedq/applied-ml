import argparse

import matplotlib
import pandas as pd

from src import ablation, calibration, eda, evaluation, explain, fairness, imbalance, tuning
from src.data import TARGET, age_group, features_and_target, load_data, split_data
from src.models import MODEL_NAMES, build_pipeline, predict
from src.utils import load_config, load_json, load_model, save_figure, save_json, save_model, save_table

matplotlib.use("Agg")


def run_eda(config, splits):
    train = splits["train"]
    numeric, categorical = eda.column_types(train)
    with_groups = train.assign(age_group=age_group(train, config))
    rates = eda.subscription_rate_table(with_groups, categorical + ["age_group"])
    drift = eda.time_drift_table(train)
    save_json({
        "train_rows": len(train),
        "features": train.shape[1] - 1,
        "duplicated_rows": int(train.duplicated().sum()),
        "subscription_rate": train[TARGET].mean(),
        "never_contacted_share": (train["pdays"] == -1).mean(),
    }, "eda_summary")
    save_table(eda.overview(train), "eda_overview")
    save_table(eda.numeric_summary(train, numeric), "eda_numeric")
    save_table(rates, "eda_categorical")
    save_table(drift, "eda_time_drift", index=False)
    save_figure(eda.plot_target(train), "eda_target")
    save_figure(eda.plot_numeric(train, numeric), "eda_numeric")
    save_figure(eda.plot_subscription_rate(rates, train[TARGET].mean()), "eda_categorical")
    save_figure(eda.plot_correlation(train, numeric), "eda_correlation")
    save_figure(eda.plot_time_drift(drift), "eda_time_drift")


def run_imbalance(config, splits):
    X, y = features_and_target(splits["train"])
    table = imbalance.compare_class_weights(X, y, config)
    save_table(table, "imbalance_class_weights", index=False)
    print(table.round(3).to_string(index=False))


def run_screening(config, splits):
    X, y = features_and_target(splits["train"])
    table = tuning.screen_models(X, y, config)
    save_table(table, "screening_models", index=False)
    print(table.round(3).to_string(index=False))


def run_tune(config, splits):
    X, y = features_and_target(splits["train"])
    studies = {name: tuning.tune(name, X, y, config) for name in MODEL_NAMES}
    run_log = pd.concat([tuning.trials_table(study, name) for name, study in studies.items()])
    save_table(run_log, "tuning_run_log", index=False)
    save_json({name: study.best_params for name, study in studies.items()}, "tuning_best_params")
    save_figure(tuning.plot_history(run_log), tuning.HISTORY_FIGURE)
    print(run_log[run_log["is_best"]].round(4).to_string(index=False))


def run_ablation(config, splits):
    best_params = load_json("tuning_best_params")
    features = ablation.feature_ablation(splits, best_params, config)
    time_proxies = ablation.time_proxy_ablation(splits, best_params, config)
    save_table(features, "ablation_features", index=False)
    save_table(time_proxies, "ablation_time_proxies", index=False)
    print(features.round(4).to_string(index=False))
    print(time_proxies.round(4).to_string(index=False))


def run_train(config, splits):
    X, y = features_and_target(splits["train"])
    best_params = load_json("tuning_best_params")
    save_model(build_pipeline("baseline", {}, config["seed"]).fit(X, y), "baseline")
    for name in MODEL_NAMES:
        save_model(build_pipeline(name, best_params[name], config["seed"]).fit(X, y), name)


def run_evaluate(config, splits):
    X_val, y_val = features_and_target(splits["validation"])
    X_test, y_test = features_and_target(splits["test"])
    models = {name: load_model(name) for name in ["baseline", *MODEL_NAMES]}
    val_proba = {name: predict(model, X_val) for name, model in models.items()}
    test_proba = {name: predict(model, X_test) for name, model in models.items()}
    thresholds = {name: evaluation.best_threshold(y_val, proba) for name, proba in val_proba.items()}
    validation = evaluation.metrics_table(y_val, val_proba, thresholds)
    test = evaluation.metrics_table(y_test, test_proba, thresholds)
    final = validation.loc[MODEL_NAMES, "average_precision"].idxmax()
    threshold = thresholds[final]

    intervals = evaluation.bootstrap_intervals(
        y_test, {name: test_proba[name] for name in MODEL_NAMES}, final,
        config["bootstrap"]["n_resamples"], config["seed"],
    )
    save_table(validation, "evaluation_validation")
    save_table(test, "evaluation_test")
    save_table(intervals, "evaluation_test_intervals")
    save_model({"name": final, "pipeline": models[final], "threshold": threshold}, "final_model")
    save_json({"final_model": final, "threshold": threshold}, "evaluation_selection")
    model_proba = {name: val_proba[name] for name in MODEL_NAMES}
    save_figure(evaluation.plot_curves(y_val, model_proba), "evaluation_curves")
    save_figure(evaluation.plot_threshold(y_val, val_proba[final], threshold), "evaluation_threshold")
    print(test.round(3).to_string())
    print(f"Final model: {final} (threshold = {threshold:.3f})")


def run_time_split(config, splits):
    final = load_model("final_model")["name"]
    X_val, y_val = features_and_target(splits["validation"])
    random_proba = {name: predict(load_model(name), X_val) for name in MODEL_NAMES}
    oldest, y_new, time_proba = evaluation.time_split_predictions(
        splits, load_json("tuning_best_params"), config["seed"]
    )
    table = pd.concat({
        "random": evaluation.split_scores(splits["train"], y_val, random_proba),
        "time": evaluation.split_scores(oldest, y_new, time_proba),
    }, names=["split"])
    intervals = evaluation.bootstrap_intervals(
        y_new, time_proba, final, config["bootstrap"]["n_resamples"], config["seed"]
    )
    save_table(table, "time_split_check")
    save_table(intervals, "time_split_intervals")
    print(table.round(3).to_string())
    print(intervals.round(3).to_string())


def run_calibration(config, splits):
    final = load_model("final_model")
    tables, summary, probas = {}, {}, {}
    for split in ["validation", "test"]:
        X, y = features_and_target(splits[split])
        probas[split] = predict(final["pipeline"], X)
        tables[split] = calibration.calibration_table(y, probas[split])
        summary[split] = calibration.summarize(y, probas[split], tables[split])
        save_table(tables[split], f"calibration_{split}")
    save_json(summary, "calibration_summary")
    save_figure(calibration.plot_reliability(tables, probas["test"], "test"), "calibration_curve")
    print(pd.DataFrame(summary).round(4).to_string())


def run_explain(config, splits):
    final = load_model("final_model")
    name, pipeline = final["name"], final["pipeline"]
    unit = explain.SHAP_UNITS[name]
    X_train, _ = features_and_target(splits["train"])
    X_test, y_test = features_and_target(splits["test"])
    settings, seed = config["shap"], config["seed"]
    sample = X_test.sample(settings["n_explained"], random_state=seed)
    background = X_train.sample(settings["n_background"], random_state=seed)
    shap_values = explain.compute_shap_values(pipeline, name, background, sample)
    importance = explain.global_importance(shap_values)
    proba = predict(pipeline, sample)
    cases = explain.pick_cases(y_test.loc[sample.index], proba, final["threshold"])

    save_table(importance, "shap_importance")
    save_table(explain.case_table(shap_values, cases, proba, unit), "shap_cases", index=False)
    save_figure(explain.plot_beeswarm(shap_values, unit), "shap_beeswarm")
    for case, position in cases.items():
        title = (
            f"{case.replace('_', ' ').capitalize()}: predicted probability = {proba[position]:.2f}, "
            f"SHAP values in {unit}"
        )
        save_figure(explain.plot_waterfall(shap_values[position], title), f"shap_{case}")
    print(importance.head(10).round(3).to_string())


def run_fairness(config, splits):
    final = load_model("final_model")
    pipeline, threshold = final["pipeline"], final["threshold"]
    X_val, y_val = features_and_target(splits["validation"])
    X_test, y_test = features_and_target(splits["test"])
    groups_val = age_group(splits["validation"], config)
    groups_test = age_group(splits["test"], config)
    group_thresholds = fairness.equal_opportunity_thresholds(
        y_val, predict(pipeline, X_val), groups_val, threshold
    )
    test_proba = predict(pipeline, X_test)
    predictions = {
        "before": (test_proba >= threshold).astype(int),
        "after": (test_proba >= groups_test.map(group_thresholds).to_numpy()).astype(int),
    }
    tables = {version: fairness.group_table(y_test, pred, groups_test) for version, pred in predictions.items()}
    summary = pd.DataFrame({
        version: fairness.fairness_summary(y_test, predictions[version], table)
        for version, table in tables.items()
    }).T.rename_axis("version")

    save_json({"single_threshold": threshold, "group_thresholds": group_thresholds}, "fairness_thresholds")
    save_table(pd.concat(tables, names=["version"]), "fairness_groups")
    save_table(summary, "fairness_summary")
    save_figure(fairness.plot_groups(tables), "fairness_groups")
    print(summary.round(3).to_string())


STEPS = {
    "eda": run_eda,
    "imbalance": run_imbalance,
    "screening": run_screening,
    "tune": run_tune,
    "ablation": run_ablation,
    "train": run_train,
    "evaluate": run_evaluate,
    "time_split": run_time_split,
    "calibration": run_calibration,
    "explain": run_explain,
    "fairness": run_fairness,
}


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the term deposit model.")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS), help="steps to run")
    args = parser.parse_args()
    config = load_config()
    splits = split_data(load_data(config), config)
    for step, run in STEPS.items():
        if step in args.steps:
            print(f"--- {step}")
            run(config, splits)


if __name__ == "__main__":
    main()
