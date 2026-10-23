"""Native three-class XGBoost, validation-only early stopping and named importance."""

import numpy as np
import xgboost as xgb


def fit_tree(train_x, train_y, validation_x, validation_y, names, config):
    dtrain = xgb.DMatrix(train_x, label=train_y + 1, feature_names=names, nthread=1)
    parameters = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cpu",
        "nthread": 1,
        "seed": config.seed,
        "max_depth": config.max_depth,
        "eta": config.tree_learning_rate,
        "min_child_weight": config.min_child_weight,
        "lambda": config.reg_lambda,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "base_score": 0.5,
    }
    evaluations = [(dtrain, "train")]
    callbacks = []
    if len(validation_y):
        dval = xgb.DMatrix(validation_x, label=validation_y + 1, feature_names=names, nthread=1)
        evaluations.append((dval, "validation"))
        callbacks.append(
            xgb.callback.EarlyStopping(
                rounds=config.patience,
                metric_name="mlogloss",
                data_name="validation",
                maximize=False,
                save_best=True,
                min_delta=config.min_delta,
            )
        )
    history = {}
    booster = xgb.train(
        parameters,
        dtrain,
        num_boost_round=config.rounds,
        evals=evaluations,
        evals_result=history,
        callbacks=callbacks,
        verbose_eval=False,
    )
    selected = booster.num_boosted_rounds()
    report = {
        "status": "fitted",
        "backend": "xgboost",
        "parameters": parameters,
        "rounds_executed": len(history["train"]["mlogloss"]),
        "selected_rounds": selected,
        "best_iteration_zero_based": selected - 1,
        "best_validation_loss": float(booster.best_score) if len(validation_y) else None,
        "selection": "validation_mlogloss" if len(validation_y) else "fixed_rounds_no_validation",
        "history": history,
    }
    return booster, report


def predict_tree(booster, matrix, names):
    return booster.predict(xgb.DMatrix(matrix, feature_names=names, nthread=1))


def importance(booster, names):
    metrics = {
        kind: booster.get_score(importance_type=kind)
        for kind in ("weight", "gain", "total_gain", "cover")
    }
    if any(set(values) - set(names) for values in metrics.values()):
        raise ValueError("tree importance schema mismatch")
    total = sum(metrics["total_gain"].values())
    return {
        "feature_columns": names,
        "method": "native split statistics, not causal effects",
        "rows": [
            {
                "feature": name,
                **{k: float(v.get(name, 0)) for k, v in metrics.items()},
                "normalized_total_gain": metrics["total_gain"].get(name, 0) / total
                if total
                else 0.0,
            }
            for name in names
        ],
    }


def contributions(booster, matrix, names):
    data = xgb.DMatrix(matrix, feature_names=names, nthread=1)
    values = booster.predict(data, pred_contribs=True, strict_shape=True)
    margins = booster.predict(data, output_margin=True, strict_shape=True)
    np.testing.assert_allclose(values.sum(axis=2), margins, rtol=1e-5, atol=1e-5)
    return {
        "kind": "native_tree_shap",
        "scale": "raw class margin before softmax",
        "class_order": [-1, 0, 1],
        "feature_columns_plus_bias": names + ["bias"],
        "values": values.tolist(),
        "margins": margins.tolist(),
        "max_additivity_error": float(np.max(np.abs(values.sum(axis=2) - margins))),
    }
