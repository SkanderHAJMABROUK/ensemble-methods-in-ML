import numpy as np
import pandas as pd
import time
import os
import copy
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

from prepdata import data_recovery

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons, make_swiss_roll
from sklearn.utils import check_random_state

from scipy import optimize
import matplotlib
matplotlib.use('Agg')  # non-interactif (pas besoin d'affichage graphique)
import matplotlib.pyplot as plt

# Librairies optionnelles
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    XGBClassifier = None
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    LGBMClassifier = None
    HAS_LGBM = False


#  Utilitaires


def imbalance_ratio(y):
    """Ratio majoritaire / minoritaire."""
    c = Counter(y)
    if len(c) == 1:
        return np.inf
    return max(c.values()) / min(c.values())


def oversample_minority(X, y, random_state=0):
    """Over-sampling aléatoire de la classe minoritaire."""
    X, y = np.asarray(X), np.asarray(y)
    counts = Counter(y)
    max_count = max(counts.values())
    rng = np.random.RandomState(random_state)
    Xs, ys = [], []
    for cls, cnt in counts.items():
        idx = np.where(y == cls)[0]
        if cnt < max_count:
            extra = rng.choice(idx, size=(max_count - cnt), replace=True)
            idx = np.concatenate([idx, extra])
        Xs.append(X[idx]);  ys.append(y[idx])
    X_new = np.vstack(Xs);  y_new = np.hstack(ys)
    perm = rng.permutation(len(y_new))
    return X_new[perm], y_new[perm]


def assign_group(y):
    """
    Groupe A : dataset balancé (IR < 3) et binaire.
    Groupe B : dataset déséquilibré (IR >= 3) OU multiclasse.
    Le seuil 3 correspond à ~75/25 split, standard dans la littérature.
    """
    ir = imbalance_ratio(y)
    n_classes = len(np.unique(y))
    if n_classes == 2 and ir < 3.0:
        return 'A', ir
    return 'B', ir


#  Algorithme 1 : Gradient Boosting avec Random Fourier Features

class RFFGradientBoosting(BaseEstimator, ClassifierMixin):
    """
    Classifieur binaire implementant l'Algorithme 1 de l'énoncé.

    À chaque itération t :
      - w_i   = exp(-y_i * H_{t-1}(x_i))           [poids exponentiels]
      - ỹ_i   = y_i * w_i                           [pseudo-résidus]
      - ω  tiré de N(0, 2γ)^d  (initialisation)
      - x^t   = argmin  (1/n) Σ exp(-ỹ_i cos(ω·(x_i - x^t)))    [step 6]
      - ω^t   = argmin  λ‖ω‖² + (1/n) Σ exp(-ỹ_i cos(ω·(x_i-x^t)))  [step 7]
      - α^t   = (1/2) ln(Σ(1+y cos(ω^t·(x_i-x^t)))w / Σ(1-y …)w)    [step 8]
      - H_t   = H_{t-1} + α^t cos(ω^t·(x_i-x^t))                [step 9]

    Pourquoi c'est du boosting :
      Les w_i concentrent le poids sur les exemples mal classés (comme AdaBoost).
      On apprend un nouvel « apprenants faible » (landmark + fréquence cosinus)
      qui minimise l'exponentielle de la perte sur les résidus courants.
      Le modèle final est une somme pondérée de fonctions cosinus : classifieur additif.
    """

    def __init__(self, T=50, gamma=1.0, lambda_reg=1.0,
                 maxiter_opt=200, random_state=None, verbose=0):
        self.T = int(T)
        self.gamma = float(gamma)
        self.lambda_reg = float(lambda_reg)
        self.maxiter_opt = int(maxiter_opt)
        self.random_state = random_state
        self.verbose = verbose

    def _to_pm1(self, y):
        """Convertit des labels {0,1} ou {cls0,cls1} en {-1, +1}."""
        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError("RFFGradientBoosting : classification binaire uniquement.")
        self.classes_ = classes
        return np.where(y == classes[1], 1.0, -1.0)

    def _init_H0(self, y):
        """
        H0 = (1/2) ln(#{y=+1} / #{y=-1})
        Initialisation de l'Algorithme 1, ligne 1.
        """
        pos = max(np.sum(y == 1.0), 1e-12)
        neg = max(np.sum(y == -1.0), 1e-12)
        return 0.5 * np.log(pos / neg)

    # Fonctions objectif avec gradient analytique

    def _obj_xt(self, x_flat, omega, X, y_tilde):
        """
        Objectif step 6 : min_{x^t} (1/n) Σ exp(-ỹ_i cos(ω·(x_i - x^t)))
        Gradient analytique par rapport à x^t.
        """
        x = x_flat.reshape(-1)
        prods = X.dot(omega) - x.dot(omega)          # ω·(x_i - x^t)
        exps  = np.exp(-y_tilde * np.cos(prods))
        fval  = np.mean(exps)
        # ∂f/∂x^t = -(1/n) Σ ỹ_i sin(ω·(x_i-x^t)) exp(…) · ω
        coeff = -np.mean(y_tilde * np.sin(prods) * exps)
        grad  = coeff * omega
        return fval, grad

    def _obj_omega(self, omega_flat, xt, X, y_tilde):
        """
        Objectif step 7 : min_ω λ‖ω‖² + (1/n) Σ exp(-ỹ_i cos(ω·(x_i - x^t)))
        Gradient analytique par rapport à ω.
        """
        omega = omega_flat.reshape(-1)
        diff  = X - xt                                # (n, d)
        prods = diff.dot(omega)                       # ω·(x_i - x^t)
        exps  = np.exp(-y_tilde * np.cos(prods))
        fval  = self.lambda_reg * np.sum(omega ** 2) + np.mean(exps)
        # ∂f/∂ω = 2λω + (1/n) Σ ỹ_i sin(ω·Δx_i) exp(…) · Δx_i
        tmp  = (y_tilde * np.sin(prods) * exps)[:, None] * diff
        grad = 2.0 * self.lambda_reg * omega + np.mean(tmp, axis=0)
        return fval, grad

    # Entraînement

    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        X   = np.asarray(X, dtype=float)
        y   = self._to_pm1(np.asarray(y))
        n, d = X.shape

        H = np.full(n, self._init_H0(y))
        self.H0_ = float(H[0])
        self.omegas_ = [];  self.xts_ = [];  self.alphas_ = []

        for t in range(1, self.T + 1):
            if self.verbose:
                print(f"  [RFF] itération {t}/{self.T}")

            # Step 3–4 : poids et pseudo-résidus
            w       = np.exp(-y * H)
            y_tilde = y * w

            # Step 5 : tirage initial de ω
            omega0 = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(d,))

            # Step 6 : optimisation de x^t
            # NOTE : on passe la fonction retournant (fval, grad) directement.
            # fmin_l_bfgs_b accepte ce format et ne calcule qu'un seul appel par éval.
            try:
                xt_opt, _, _ = optimize.fmin_l_bfgs_b(
                    self._obj_xt,
                    x0=X.mean(axis=0),
                    args=(omega0, X, y_tilde),
                    maxiter=self.maxiter_opt
                )
            except Exception:
                xt_opt = X.mean(axis=0)
            xt_opt = np.asarray(xt_opt).reshape(d)

            # Step 7 : optimisation de ω^t
            try:
                omega_opt, _, _ = optimize.fmin_l_bfgs_b(
                    self._obj_omega,
                    x0=omega0,
                    args=(xt_opt, X, y_tilde),
                    maxiter=self.maxiter_opt
                )
            except Exception:
                omega_opt = omega0
            omega_opt = np.asarray(omega_opt).reshape(d)

            # Step 8 : calcul de α^t
            vals = np.cos((X - xt_opt).dot(omega_opt))
            num  = max(np.sum((1.0 + y * vals) * w), 1e-12)
            den  = max(np.sum((1.0 - y * vals) * w), 1e-12)
            alpha_t = 0.5 * np.log(num / den)

            # Step 9 : mise à jour de H
            H += alpha_t * vals

            self.omegas_.append(omega_opt.copy())
            self.xts_.append(xt_opt.copy())
            self.alphas_.append(float(alpha_t))

        return self

    # Prédiction

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        H = np.full(X.shape[0], self.H0_, dtype=float)
        for omega, xt, alpha in zip(self.omegas_, self.xts_, self.alphas_):
            H += alpha * np.cos((X - xt).dot(omega))
        return H

    def predict(self, X):
        H = self.decision_function(X)
        return np.where(H >= 0.0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        H = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-2.0 * H))
        return np.vstack([1.0 - p, p]).T


#  Wrapper One-vs-Rest pour la classification multiclasse

class OneVsRestRFF(BaseEstimator, ClassifierMixin):
    """Encapsule RFFGradientBoosting dans un schéma One-vs-Rest."""

    def __init__(self, base_estimator=None):
        self.base_estimator = (base_estimator
                               if base_estimator is not None
                               else RFFGradientBoosting())

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        self.models_ = {}
        for cls in self.classes_:
            y_bin = np.where(y == cls, 1.0, -1.0)
            clf = clone(self.base_estimator)
            clf.fit(X, y_bin)
            self.models_[cls] = clf
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            scores[:, i] = self.models_[cls].decision_function(X)
        return scores

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


#  Pipeline d'évaluation (Section 2)

def build_algorithms():
    """Retourne le dictionnaire des algorithmes à comparer."""
    base_estimators = [
        ('svm',  SVC(kernel='linear', probability=True, max_iter=2000)),
        ('knn',  KNeighborsClassifier(5)),
        ('rf',   RandomForestClassifier(n_estimators=50, random_state=0)),
    ]
    algos = {
        'LogReg':       lambda: LogisticRegression(C=1.0, max_iter=1000,
                                                   solver='lbfgs'),
        'Bagging':      lambda: BaggingClassifier(n_estimators=50,
                                                  random_state=0),
        'RandomForest': lambda: RandomForestClassifier(n_estimators=100,
                                                       random_state=0),
        'AdaBoost':     lambda: AdaBoostClassifier(n_estimators=50,
                                                   random_state=0,
                                                   algorithm='SAMME'),
        'GradBoost':    lambda: GradientBoostingClassifier(n_estimators=100,
                                                           random_state=0),
        'Stacking':     lambda: StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3
        ),
    }
    if HAS_XGB:
        algos['XGBoost'] = lambda: XGBClassifier(
            n_estimators=100, eval_metric='logloss',
            use_label_encoder=False, verbosity=0, random_state=0
        )
    if HAS_LGBM:
        algos['LGBM'] = lambda: LGBMClassifier(
            n_estimators=100, verbose=-1, random_state=0
        )
    algos['RFF_binary']  = lambda: RFFGradientBoosting(
        T=30, gamma=1.0, lambda_reg=1.0, random_state=0
    )
    algos['RFF_OvR']     = lambda: OneVsRestRFF(
        RFFGradientBoosting(T=30, gamma=1.0, lambda_reg=1.0, random_state=0)
    )
    return algos


def evaluate_model_cv(model, X, y, skf, group):
    """
    Évalue un modèle par validation croisée stratifiée.
    Groupe A (balancé)   → F1 macro
    Groupe B (déséquilibré / multiclasse) → F1 binaire sur classe minoritaire
    """
    accs, f1s, times = [], [], []
    # Pour groupe B binaire : identifier la classe minoritaire
    counts = Counter(y)
    minority_cls = min(counts, key=counts.get) if group == 'B' else None

    for tr_idx, val_idx in skf.split(X, y):
        Xtr, Xv = X[tr_idx], X[val_idx]
        ytr, yv = y[tr_idx], y[val_idx]
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xv  = scaler.transform(Xv)

        t0 = time.time()
        model.fit(Xtr, ytr)
        times.append(time.time() - t0)

        yp = model.predict(Xv)
        accs.append(accuracy_score(yv, yp))
        if group == 'B' and len(np.unique(y)) == 2:
            # F1 sur la classe minoritaire (plus informatif pour déséquilibré)
            f1s.append(f1_score(yv, yp, pos_label=minority_cls,
                                average='binary', zero_division=0))
        else:
            f1s.append(f1_score(yv, yp, average='macro', zero_division=0))

    return (np.mean(accs), np.std(accs),
            np.mean(f1s),  np.std(f1s),
            np.mean(times))


def run_section2(datasets_list, out_csv='results_section2.csv'):
    """
    Section 2 : Comparaison complète sur tous les datasets.
    Pour le groupe B, teste aussi l'oversampling et le cost-sensitive learning.
    """
    summary = []
    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    algos   = build_algorithms()

    for ds in datasets_list:
        print(f"\n{'='*60}\nDataset : {ds}")
        try:
            X, y = data_recovery(ds)
        except Exception as e:
            print(f"  !! Impossible de charger {ds} : {e}")
            continue
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n_classes = len(np.unique(y))
        group, ir = assign_group(y)
        print(f"  → Groupe {group}  |  IR={ir:.1f}  |  classes={n_classes}")

        modes = ['vanilla']
        if group == 'B':
            modes += ['oversample', 'cost_sensitive']

        for mode in modes:
            Xm, ym = (oversample_minority(X, y, 42)
                      if mode == 'oversample'
                      else (X.copy(), y.copy()))

            for name, mk in algos.items():
                if name == 'RFF_binary' and n_classes != 2:
                    continue   # binaire seulement
                if name == 'RFF_OvR' and n_classes == 2:
                    continue   # OvR inutile si binaire

                model = mk()

                # Appliquer class_weight='balanced' si disponible
                if mode == 'cost_sensitive':
                    try:
                        model.set_params(class_weight='balanced')
                    except Exception:
                        pass

                try:
                    res_tuple = evaluate_model_cv(model, Xm, ym, skf, group)
                except Exception as e:
                    print(f"    ✗ {name} [{mode}] : {e}")
                    continue

                acc_m, acc_s, f1_m, f1_s, t_m = res_tuple
                row = {
                    'Dataset': ds, 'Group': group, 'Mode': mode,
                    'Algorithm': name,
                    'Accuracy_mean': round(acc_m, 4),
                    'Accuracy_std':  round(acc_s, 4),
                    'F1_mean': round(f1_m, 4),
                    'F1_std':  round(f1_s, 4),
                    'Time_s':  round(t_m,  3),
                    'n_samples': len(ym),
                    'n_classes': n_classes,
                    'IR': round(ir, 2),
                }
                summary.append(row)
                print(f"    {name:22s} [{mode:14s}]  acc={acc_m:.4f}  "
                      f"f1={f1_m:.4f}  t={t_m:.2f}s")

    df = pd.DataFrame(summary)
    df.to_csv(out_csv, index=False)
    print(f"\n[Section 2] Résultats sauvegardés → {out_csv}")
    return df


#  Section 3 — Expériences visuelles et comparatives

def _plot_boundaries(models_dict, X, y, suptitle, filename):
    """Trace les frontières de décision pour chaque modèle."""
    n = len(models_dict)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = np.array(axs).reshape(-1)

    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    for i, (name, model) in enumerate(models_dict.items()):
        ax = axs[i]
        try:
            Z = model.decision_function(grid)
            if Z.ndim > 1:
                Z = Z[:, 1] - Z[:, 0]
        except Exception:
            Z = model.predict(grid).astype(float)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=30)
        ax.set_title(name, fontsize=11)
        ax.set_xticks([]);  ax.set_yticks([])

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → Figure sauvegardée : {filename}")


def demo_few_points_iterations():
    """
    Exigence §3.3, point 1 :
    Montrer la frontière de décision aux 3 premières itérations
    sur un dataset 2D avec peu de points.
    """
    print("\n[Demo] Frontières aux 3 premières itérations (peu de points)")
    X, y = make_moons(n_samples=60, noise=0.2, random_state=7)

    # Entraîner jusqu'à T=6 pour avoir au moins 3 itérations stables
    rff = RFFGradientBoosting(T=6, gamma=1.0, lambda_reg=1.0,
                              random_state=0, verbose=1)
    rff.fit(X, y)

    models = {}
    for k in range(1, 4):
        m = copy.deepcopy(rff)
        m.omegas_ = m.omegas_[:k]
        m.xts_    = m.xts_[:k]
        m.alphas_ = m.alphas_[:k]
        models[f'RFF — itération {k}'] = m

    _plot_boundaries(models, X, y,
                     suptitle='RFF-GB : évolution frontière (dataset 60 points)',
                     filename='fig_few_points_iterations.png')


def demo_toy_comparison():
    """
    Exigence §3.3, point 2 :
    Comparer RFF, XGBoost, Kernel SVM sur make_moons et make_swiss_roll.
    """
    print("\n[Demo] Comparaison frontières — Moons + Swiss Roll")

    # Make Moons
    X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                           random_state=0, stratify=y)
    models_moons = {}
    rff = RFFGradientBoosting(T=30, gamma=1.0, lambda_reg=1.0,
                              random_state=0, verbose=0)
    rff.fit(Xtr, ytr)
    models_moons['RFF-GB (T=30)'] = rff

    svc = SVC(kernel='rbf', gamma='scale', random_state=0).fit(Xtr, ytr)
    models_moons['Kernel SVM (RBF)'] = svc

    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=100, eval_metric='logloss',
                             use_label_encoder=False, verbosity=0,
                             random_state=0)
        xgb.fit(Xtr, ytr)
        models_moons['XGBoost'] = xgb

    _plot_boundaries(models_moons, Xtr, ytr,
                     suptitle='Comparaison frontières — Make Moons',
                     filename='fig_moons_comparison.png')

    # Swiss Roll (projection 2D) 
    X3D, t_val = make_swiss_roll(n_samples=600, noise=0.2, random_state=0)
    X2D  = X3D[:, [0, 2]]
    # Binarisation : moitié inférieure / supérieure du paramètre t
    y_sr = (t_val > np.median(t_val)).astype(int)

    Xtr2, Xte2, ytr2, yte2 = train_test_split(X2D, y_sr, test_size=0.3,
                                               random_state=0, stratify=y_sr)
    models_sr = {}
    rff2 = RFFGradientBoosting(T=30, gamma=1.0, lambda_reg=1.0,
                               random_state=0, verbose=0)
    rff2.fit(Xtr2, ytr2)
    models_sr['RFF-GB (T=30)'] = rff2

    svc2 = SVC(kernel='rbf', gamma='scale', random_state=0).fit(Xtr2, ytr2)
    models_sr['Kernel SVM (RBF)'] = svc2

    if HAS_XGB:
        xgb2 = XGBClassifier(n_estimators=100, eval_metric='logloss',
                              use_label_encoder=False, verbosity=0,
                              random_state=0)
        xgb2.fit(Xtr2, ytr2)
        models_sr['XGBoost'] = xgb2

    _plot_boundaries(models_sr, Xtr2, ytr2,
                     suptitle='Comparaison frontières — Swiss Roll (proj. 2D)',
                     filename='fig_swissroll_comparison.png')


def demo_perf_vs_iterations(dataset='iono'):
    """
    Exigence §3.3, point 3 :
    Comparer performance ET temps en fonction du nombre d'itérations
    pour RFF, XGBoost, LGBM. Kernel SVM comme baseline.
    """
    print(f"\n[Demo] Performance vs itérations — dataset : {dataset}")

    try:
        X, y = data_recovery(dataset)
        X, y = np.asarray(X, dtype=float), np.asarray(y)
    except Exception as e:
        print(f"  !! Impossible de charger {dataset} : {e}")
        return

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                           random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    iter_range = [5, 10, 20, 30, 50, 75, 100]
    results = {}   # {method: {'f1': [], 'time': []}}

    # RFF
    results['RFF-GB'] = {'f1': [], 'time': []}
    for T in iter_range:
        t0  = time.time()
        clf = RFFGradientBoosting(T=T, gamma=1.0, lambda_reg=1.0,
                                  random_state=0)
        clf.fit(Xtr, ytr)
        elapsed = time.time() - t0
        f1 = f1_score(yte, clf.predict(Xte), average='macro',
                      zero_division=0)
        results['RFF-GB']['f1'].append(f1)
        results['RFF-GB']['time'].append(elapsed)
        print(f"  RFF  T={T:3d}  f1={f1:.4f}  t={elapsed:.2f}s")

    # XGBoost
    if HAS_XGB:
        results['XGBoost'] = {'f1': [], 'time': []}
        for T in iter_range:
            t0  = time.time()
            clf = XGBClassifier(n_estimators=T, eval_metric='logloss',
                                use_label_encoder=False, verbosity=0,
                                random_state=0)
            clf.fit(Xtr, ytr)
            elapsed = time.time() - t0
            f1 = f1_score(yte, clf.predict(Xte), average='macro',
                          zero_division=0)
            results['XGBoost']['f1'].append(f1)
            results['XGBoost']['time'].append(elapsed)

    # LGBM
    if HAS_LGBM:
        results['LGBM'] = {'f1': [], 'time': []}
        for T in iter_range:
            t0  = time.time()
            clf = LGBMClassifier(n_estimators=T, verbose=-1, random_state=0)
            clf.fit(Xtr, ytr)
            elapsed = time.time() - t0
            f1 = f1_score(yte, clf.predict(Xte), average='macro',
                          zero_division=0)
            results['LGBM']['f1'].append(f1)
            results['LGBM']['time'].append(elapsed)

    # Kernel SVM baseline (ligne horizontale) 
    t0  = time.time()
    svc = SVC(kernel='rbf', gamma='scale', random_state=0)
    svc.fit(Xtr, ytr)
    svm_time = time.time() - t0
    svm_f1   = f1_score(yte, svc.predict(Xte), average='macro',
                        zero_division=0)
    print(f"  SVM baseline  f1={svm_f1:.4f}  t={svm_time:.2f}s")

    # Tracé
    colors = {'RFF-GB': '#E24B4A', 'XGBoost': '#185FA5', 'LGBM': '#1D9E75'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for method, data in results.items():
        c = colors.get(method, 'gray')
        ax1.plot(iter_range, data['f1'], marker='o', label=method, color=c)
        ax2.plot(iter_range, data['time'], marker='s', label=method, color=c)

    ax1.axhline(svm_f1, linestyle='--', color='#888780', alpha=0.7,
                label=f'SVM RBF ({svm_f1:.3f})')
    ax1.set_xlabel('Nombre d\'itérations')
    ax1.set_ylabel('F1-score (macro)')
    ax1.set_title(f'Performance vs itérations — {dataset}')
    ax1.legend(fontsize=9);  ax1.grid(alpha=0.3)

    ax2.axhline(svm_time, linestyle='--', color='#888780', alpha=0.7,
                label=f'SVM RBF ({svm_time:.2f}s)')
    ax2.set_xlabel('Nombre d\'itérations')
    ax2.set_ylabel('Temps d\'entraînement (s)')
    ax2.set_title(f'Temps d\'entraînement vs itérations — {dataset}')
    ax2.legend(fontsize=9);  ax2.grid(alpha=0.3)

    plt.tight_layout()
    fname = f'fig_perf_vs_iters_{dataset}.png'
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → Figure sauvegardée : {fname}")


#  Point d'entrée principal

# Liste des 25 datasets disponibles dans prepdata.py
DATASETS_ALL = [
    # Très déséquilibrés (Groupe B attendu)
    'abalone8', 'abalone17', 'abalone20', 'wine4', 'yeast6',
    # Modérément déséquilibrés (Groupe B)
    'autompg', 'glass', 'hayes', 'libras', 'pageblocks',
    'segmentation', 'vehicle',
    # Relativement balancés (Groupe A ou B selon IR)
    'australian', 'balance', 'bupa', 'german', 'heart',
    'iono', 'newthyroid', 'pima', 'sonar', 'spambase',
    'wine', 'wdbc', 'yeast3',
]


def main():
    print("=" * 60)
    print("  ENSEMBLE METHODS — Projet M2 MALIA")
    print("=" * 60)

    # Section 2 : Comparaison sur tous les datasets
    print("\n>>> SECTION 2 : Comparaison des algorithmes")
    results_df = run_section2(DATASETS_ALL, out_csv='results_section2.csv')

    # Section 3 : Expériences RFF
    print("\n>>> SECTION 3 : Gradient Boosting avec RFF")

    # 3.3 — Point 1 : frontières sur 3 premières itérations (peu de points)
    demo_few_points_iterations()

    # 3.3 — Point 2 : comparaison RFF vs XGBoost vs Kernel SVM
    demo_toy_comparison()

    # 3.3 — Point 3 : performance + temps vs nb itérations
    # (utilise 'iono' par défaut ; changer selon votre préférence)
    demo_perf_vs_iterations(dataset='iono')

    print("\n[main] Terminé. Tous les résultats et figures sont sauvegardés.")


if __name__ == '__main__':
    main()
