"""Baseline, LightGBM, dan XGBoost multi-horizon.

Jalankan dari folder `extractions`:
    ..\\.venv\\Scripts\\python.exe -m forecast.model_gbdt [--trials 40]

Tahap:
1. Tuning Optuna dengan CV rolling-origin (expanding, 4 fold) di dalam bagian latih saja, fitur tanpa
   cuaca. Baris latih fold hanya yang targetnya sebelum awal fold (purged), baris evaluasi fold adalah
   asal di dalam jendela fold.
2. Tahap "validasi": latih pada target < TRAIN_END, ramal asal di validasi (dipakai memilih ensemble).
3. Tahap "uji": latih pada target < VAL_END, ramal asal di uji. Angka akhir dari sini.
Setiap tahap dijalankan untuk tiga skenario fitur cuaca dengan hiperparameter yang sama.
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from .common import HASIL, TRAIN_END, TZ, VAL_END, load_cuaca_5menit, load_deret, metrik
from .fitur import SKENARIO, baseline, tabel

FOLDS = [pd.Timestamp(d, tz=TZ) for d in ("2026-11-15", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-20")]
SEED = 7


def tahap(tab: pd.DataFrame, latih_sebelum: pd.Timestamp, asal_mulai: pd.Timestamp, asal_akhir: pd.Timestamp,
          target_sebelum: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = tab[tab["waktu_target"] < latih_sebelum]
    m = (tab["waktu_asal"] >= asal_mulai) & (tab["waktu_asal"] < asal_akhir)
    if target_sebelum is not None:
        m &= tab["waktu_target"] < target_sebelum
    return tr, tab[m]


def fit_lgb(params: dict, tr: pd.DataFrame, fitur: list, n_iter: int, target: str = "y",
            ev: pd.DataFrame | None = None) -> lgb.Booster:
    p = {**params, "verbosity": -1, "seed": SEED, "num_threads": 8}
    dtr = lgb.Dataset(tr[fitur], tr[target], categorical_feature=["arah_kode"], free_raw_data=False)
    if ev is None:
        return lgb.train(p, dtr, n_iter)
    dev = lgb.Dataset(ev[fitur], ev[target], reference=dtr)
    return lgb.train(p, dtr, n_iter, valid_sets=[dev], callbacks=[lgb.early_stopping(100, verbose=False)])


def cv_tabs(deret, cuaca) -> list:
    out = []
    for k in range(4):
        tab = tabel(deret, cuaca, cutoff=FOLDS[k])
        out.append(tahap(tab, FOLDS[k], FOLDS[k], FOLDS[k + 1], FOLDS[k + 1]))
    return out


def tune_lgb(folds: list, fitur: list, trials: int) -> tuple[dict, int, float]:
    def obj(trial):
        params = {
            "objective": trial.suggest_categorical("objective", ["l2", "l1", "huber"]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 96, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
            "metric": "l1",
        }
        if params["objective"] == "huber":
            params["alpha"] = 500.0
        maes, iters = [], []
        for tr, ev in folds:
            b = fit_lgb(params, tr, fitur, 3000, ev=ev)
            maes.append(np.mean(np.abs(b.predict(ev[fitur], num_iteration=b.best_iteration) - ev["y"])))
            iters.append(b.best_iteration)
        trial.set_user_attr("iter", int(np.mean(iters)))
        return float(np.mean(maes))

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=trials, show_progress_bar=False)
    best = {**st.best_params, "bagging_freq": 1, "metric": "l1"}
    if best["objective"] == "huber":
        best["alpha"] = 500.0
    return best, st.best_trial.user_attrs["iter"], st.best_value


def tune_xgb(folds: list, fitur: list, trials: int) -> tuple[dict, int, float]:
    def params_of(trial):
        return {
            "objective": trial.suggest_categorical("objective", ["reg:squarederror", "reg:absoluteerror",
                                                                 "reg:pseudohubererror"]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }

    def obj(trial):
        p = params_of(trial)
        maes, iters = [], []
        for tr, ev in folds:
            m = xgb_model(p, 3000, early=True).fit(tr[fitur], tr["y"], eval_set=[(ev[fitur], ev["y"])], verbose=False)
            maes.append(np.mean(np.abs(m.predict(ev[fitur]) - ev["y"])))
            iters.append(m.best_iteration + 1)
        trial.set_user_attr("iter", int(np.mean(iters)))
        return float(np.mean(maes))

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=trials, show_progress_bar=False)
    return st.best_params, st.best_trial.user_attrs["iter"], st.best_value


def xgb_model(p: dict, n: int, early: bool = False) -> xgb.XGBRegressor:
    extra = {"huber_slope": 500.0} if p.get("objective") == "reg:pseudohubererror" else {}
    return xgb.XGBRegressor(**p, **extra, n_estimators=n, tree_method="hist", random_state=SEED, n_jobs=8,
                            eval_metric="mae", early_stopping_rounds=100 if early else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    (HASIL / "model").mkdir(parents=True, exist_ok=True)
    deret, cuaca = load_deret(), load_cuaca_5menit()
    fit0 = SKENARIO["tanpa_cuaca"]

    folds = cv_tabs(deret, cuaca)
    print("fold:", [(len(tr), len(ev)) for tr, ev in folds])
    cv_base = {k: float(np.mean([np.mean(np.abs(baseline(ev)[k] - ev["y"])) for _, ev in folds]))
               for k in ["persistence", "seasonal_naive", "profil_historis"]}
    lgb_p, lgb_n, lgb_cv = tune_lgb(folds, fit0, args.trials)
    print("LightGBM CV MAE", round(lgb_cv, 1), lgb_p, lgb_n)
    xgb_p, xgb_n, xgb_cv = tune_xgb(folds, fit0, max(10, args.trials // 2))
    print("XGBoost CV MAE", round(xgb_cv, 1), xgb_p, xgb_n)
    tuning = {"cv_mae": {**cv_base, "lightgbm": lgb_cv, "xgboost": xgb_cv},
              "lightgbm": {"params": lgb_p, "iter": lgb_n}, "xgboost": {"params": xgb_p, "iter": xgb_n},
              "fold_awal": [str(f) for f in FOLDS], "fold_baris": [[len(tr), len(ev)] for tr, ev in folds]}
    (HASIL / "tuning_gbdt.json").write_text(json.dumps(tuning, indent=1), encoding="utf-8")

    stages = {"validasi": (TRAIN_END, TRAIN_END, VAL_END, VAL_END), "uji": (VAL_END, VAL_END, pd.Timestamp.max.tz_localize(TZ), None)}
    for nama, (latih_sebelum, mulai, akhir, tsebelum) in stages.items():
        tab = tabel(deret, cuaca, cutoff=latih_sebelum)
        tab_smp = tabel(deret, cuaca, cutoff=latih_sebelum, target="smp")
        tr, ev = tahap(tab, latih_sebelum, mulai, akhir, tsebelum)
        pred = ev[["arah", "waktu_asal", "waktu_target", "h", "y"]].copy()
        pred = pd.concat([pred, baseline(ev).set_axis(ev.index)], axis=1)
        for sk, fitur in SKENARIO.items():
            b = fit_lgb(lgb_p, tr, fitur, lgb_n)
            pred[f"lightgbm_{sk}"] = b.predict(ev[fitur])
            if nama == "uji":
                b.save_model(str(HASIL / "model" / f"lightgbm_{sk}.txt"))
            m = xgb_model(xgb_p, xgb_n).fit(tr[fitur], tr["y"])
            pred[f"xgboost_{sk}"] = m.predict(ev[fitur])
        pred.to_parquet(HASIL / f"pred_{nama}.parquet", index=False)
        # target kedua: SMP/jam (LightGBM, fitur cuaca sempurna dan tanpa cuaca)
        trs, evs = tahap(tab_smp, latih_sebelum, mulai, akhir, tsebelum)
        ps = evs[["arah", "waktu_asal", "waktu_target", "h", "y"]].copy()
        ps = pd.concat([ps, baseline(evs).set_axis(evs.index)], axis=1)
        for sk in ["tanpa_cuaca", "cuaca_sempurna"]:
            ps[f"lightgbm_{sk}"] = fit_lgb(lgb_p, trs, SKENARIO[sk], lgb_n).predict(evs[SKENARIO[sk]])
        ps.to_parquet(HASIL / f"pred_smp_{nama}.parquet", index=False)
        print(nama, "baris latih", len(tr), "baris evaluasi", len(ev))
        for k in [c for c in pred.columns if c not in ("arah", "waktu_asal", "waktu_target", "h", "y")]:
            print(f"  {k:28s}", {m: round(v, 1) for m, v in metrik(pred["y"].to_numpy(), pred[k].to_numpy()).items()})


if __name__ == "__main__":
    main()
