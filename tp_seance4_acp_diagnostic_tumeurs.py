# /// script
# dependencies = [
#     "marimo",
#     "scikit-learn==1.8.0",
#     "seaborn==0.13.2",
#     "pandas",
#     "numpy",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="TP Séance 4 — ACP & Diagnostic de tumeurs",
)


@app.cell(hide_code=True)
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def imports():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100
    sns.set_theme(style="whitegrid", palette="deep")
    np.random.seed(42)
    import warnings

    warnings.filterwarnings("ignore")
    return PCA, StandardScaler, np, plt, sns


@app.cell(hide_code=True)
def header(mo):
    mo.md(r"""
    # 🔬 TP/TD — L'ACP appliquée au diagnostic de tumeurs

    **Cours : Analyse de données · 1ère année ingénieur · 2025–2026**

    **Séance 4 — Réduction de dimension · Dataset : Breast Cancer Wisconsin (Diagnostic), UCI n°17**

    L'objectif de ce TP est de vous permettre à apprendre et comprendre les concepts aborder dans le cours. Vous pouvez me partager le lien GitHub du ficher avec autorisation de le télécharger.

    **Date limite de rendu : lundi 11/07/2026 avant 23h59**

    **Mail** : mboup.djibril@ugb.edu.sn

    ---

    ## 🎬 Le contexte métier

    Vous êtes **data analyst dans un laboratoire d'anatomo-pathologie**. Pour chaque biopsie,
    un logiciel d'imagerie mesure la forme des noyaux cellulaires et produit **30 variables
    numériques** : rayon, périmètre, aire, texture, compacité, concavité, symétrie… (chacune
    déclinée en *moyenne*, *erreur-type* et *pire valeur*). Un pathologiste a établi le
    diagnostic de référence : tumeur **maligne** (cancéreuse) ou **bénigne**.

    La cheffe de service vous demande :

    > *« 30 mesures par biopsie, c'est illisible sur un compte-rendu. Peux-tu les résumer en
    > **2 ou 3 indicateurs synthétiques** ? Et est-ce que ces indicateurs **séparent** les
    > tumeurs malignes des bénignes ? »*

    C'est le rôle de l'**ACP**. Ce TP déroule la démarche complète — et vous verrez que, cette
    fois, les données ont une structure **très favorable** à la réduction de dimension.
    """)
    return


@app.cell(hide_code=True)
def objectifs(mo):
    mo.md(r"""
    ## 🎯 Objectif du TP

    Ce TP est organisé en **16 exercices**, chacun dans **sa propre section** et doté d'une
    **vérification automatique** (✅ / ❌) exécutée par du code :

    | Bloc | Exercices | Nature | Vérification |
    |------|-----------|--------|--------------|
    | **Partie A — Rappel de cours** | **1 → 12** | **QCM** sur la théorie de l'ACP (Partie 2 du cours) | choix interactif ; réponse **hachée** |
    | **Partie B — Pratique** | **13 → 16** | code Python (ACP avec scikit-learn) | test sur votre code |

    **Objectif pédagogique :** consolider d'abord la **théorie de l'ACP** (exercices 1–12),
    puis la mettre en œuvre sur un cas réel (exercices 13–16) — déterminer le nombre de
    composantes, projeter en 2D, interpréter les *loadings*, et utiliser l'ACP comme
    prétraitement d'un classifieur.

    ## 📝 Comment travailler ce TP

    - **Partie A (QCM)** : sélectionnez une réponse ; la cellule de vérification affiche
      immédiatement si elle est correcte. La bonne réponse est stockée **hachée** — impossible
      de la lire dans le code.
    - **Partie B (pratique)** : les cellules **`Données`** et **`EDA`** sont **fournies**
      (ne les modifiez pas). Les cellules **`Exercice`** contiennent un `# 👉 À COMPLÉTER` :
      remplacez le `None`, et la vérification valide votre code.
    - 🧑‍🏫 **Les corrections détaillées** (bonnes réponses + explications + corrigés de code)
      font l'objet d'un **notebook marimo séparé** :
      **`corrige_tp_seance4_acp_diagnostic_tumeurs.py`**.

    > **Prérequis** : Parties 1 et 2 de la Séance 4 (malédiction de la dimension, ACP).
    """)
    return


@app.cell(hide_code=True)
def helpers(mo):
    """Fonction de vérification des QCM — réponses comparées par hachage (pas de spoiler)."""
    import hashlib as _hashlib

    def verifier_qcm(valeur, hash_attendu):
        if valeur is None:
            return mo.callout(
                mo.md("⏳ *Sélectionnez une réponse ci-dessus pour lancer la vérification.*"),
                kind="neutral",
            )
        _h = _hashlib.sha256(valeur.strip().upper().encode()).hexdigest()[:8]
        if _h == hash_attendu:
            return mo.callout(mo.md("✅ **Bonne réponse !**"), kind="success")
        return mo.callout(
            mo.md("❌ **Réponse incorrecte.** Revoyez la Partie 2 du cours, puis consultez "
                  "la *note de correction* séparée si besoin."),
            kind="danger",
        )

    return (verifier_qcm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 📚 Partie A — Rappel de cours (Exercices 1 → 12, QCM)

    Douze QCM pour vérifier que la **théorie de l'ACP** est en place *avant* de coder.
    Pour chacun, **une seule** réponse est correcte.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✏️ Exercice 1 — Qu'est-ce qu'une composante principale ?

    Une **composante principale**, c'est :

    - **A.** une des variables d'origine, choisie pour sa forte variance
    - **B.** une combinaison *linéaire* des variables, *non corrélée* aux autres composantes, *ordonnée par variance décroissante*
    - **C.** une combinaison *non linéaire* qui maximise la séparation entre classes
    - **D.** la moyenne des variables standardisées
    """)
    return


@app.cell
def _(mo):
    ex1 = mo.ui.radio(options=["A", "B", "C", "D"], label="**Votre réponse :**")
    ex1
    return (ex1,)


@app.cell(hide_code=True)
def _(ex1, verifier_qcm):
    verifier_qcm(ex1.value, "df7e70e5")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✏️ Exercice 2 — ACP : supervisée ou non ?

    L'ACP est une méthode :

    - **A.** supervisée : elle oriente ses axes à l'aide des étiquettes
    - **B.** non supervisée : elle ne regarde jamais les étiquettes
    - **C.** semi-supervisée
    - **D.** supervisée seulement si les données sont standardisées
    """)
    return


@app.cell
def _(mo):
    ex2 = mo.ui.radio(options=["A", "B", "C", "D"], label="**Votre réponse :**")
    ex2
    return (ex2,)


@app.cell(hide_code=True)
def _(ex2, verifier_qcm):
    verifier_qcm(ex2.value, "df7e70e5")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✏️ Exercice 3 — Pourquoi standardiser avant une ACP ?

    On **standardise** les données avant une ACP parce que :

    - **A.** l'ACP *maximise la variance* : sans cela, une variable à grande échelle domine mécaniquement PC1
    - **B.** cela accélère le calcul de la SVD
    - **C.** cela rend les composantes corrélées entre elles
    - **D.** ce n'est en réalité jamais nécessaire
    """)
    return


@app.cell
def _(mo):
    ex3 = mo.ui.radio(options=["A", "B", "C", "D"], label="**Votre réponse :**")
    ex3
    return (ex3,)


@app.cell(hide_code=True)
def _(ex3, verifier_qcm):
    verifier_qcm(ex3.value, "559aead0")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✏️ Exercice 4 — La matrice de covariance des données standardisées

    Sur données **standardisées**, la matrice de covariance
    $\Sigma = \frac{1}{n-1} X_c^\top X_c$ est égale à :

    - **A.** la matrice identité
    - **B.** la matrice des loadings
    - **C.** la matrice des scores $Z$
    - **D.** la matrice de **corrélation** (diagonale = 1)
    """)
    return


@app.cell
def _(mo):
    ex4 = mo.ui.radio(options=["A", "B", "C", "D"], label="**Votre réponse :**")
    ex4
    return (ex4,)


@app.cell(hide_code=True)
def _(ex4, verifier_qcm):
    verifier_qcm(ex4.value, "3f39d5c3")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✏️ Exercice 5 — Vecteurs propres vs valeurs propres

    Dans l'ACP, la **direction** d'un axe est donnée par… et son **importance** (variance
    expliquée) par… :

    - **A.** direction = valeur propre ; importance = vecteur propre
    - **B.** direction = vecteur propre de $\Sigma$ ; importance = valeur propre $\lambda_k$
    - **C.** direction = score ; importance = loading
    - **D.** les deux sont données par la même valeur propre
    """)
    return


@app.cell
def _(mo):
    ex5 = mo.ui.radio(options=["A", "B", "C", "D"], label="**Votre réponse :**")
    ex5
    return (ex5,)


@app.cell(hide_code=True)
def _(ex5, verifier_qcm):
    verifier_qcm(ex5.value, "df7e70e5")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✏️ Exercice 6 — La direction de variance maximale

    La solution de $\mathbf{v}^* = \arg\max_{\|\mathbf{v}\|=1}\mathbf{v}^\top\Sigma\mathbf{v}$ est :

    - **A.** le vecteur propre de $\Sigma$ associé à la **plus grande** valeur propre
    - **B.** le vecteur propre associé à la **plus petite** valeur propre
    - **C.** la moyenne de tous les vecteurs propres
    - **D.** n'importe quel vecteur unitaire
    """)
    return


@app.cell
def _(mo):
    ex6 = mo.ui.radio(options=["A", "B", "C", "D"], label="**Votre réponse :**")
    ex6
    return (ex6,)


@app.cell(hide_code=True)
def _(ex6, verifier_qcm):
    verifier_qcm(ex6.value, "559aead0")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✏️ Exercice 7 — La dualité de Pearson (1901)

    Maximiser la variance projetée revient à :

    - **A.** maximiser l'erreur de reconstruction
    - **B.** minimiser le nombre de composantes
    - **C.** **minimiser la somme des distances quadratiques** points–projections (erreur de reconstruction)
    - **D.** maximiser la corrélation entre composantes
    """)
    return


@app.cell
def _(mo):
    ex7 = mo.ui.radio(options=["A", "B", "C", "D"], label="**Votre réponse :**")
    ex7
    return (ex7,)


@app.cell(hide_code=True)
def _(ex7, verifier_qcm):
    verifier_qcm(ex7.value, "6b23c0d5")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✏️ Exercice 8 — Trace et déterminant

    Pour les valeurs propres de $\Sigma$, on a **toujours** :

    - **A.** $\text{trace}(\Sigma)=\sum_k\lambda_k$ (somme des variances) et $\det(\Sigma)=\prod_k\lambda_k$
    - **B.** $\text{trace}(\Sigma)=\prod_k\lambda_k$
    - **C.** $\det(\Sigma)=\sum_k\lambda_k$
    - **D.** $\text{trace}(\Sigma)=1$ dans tous les cas
    """)
    return


@app.cell
def _(mo):
    ex8 = mo.ui.radio(options=["A", "B", "C", "D"], label="**Votre réponse :**")
    ex8
    return (ex8,)


@app.cell(hide_code=True)
def _(ex8, verifier_qcm):
    verifier_qcm(ex8.value, "559aead0")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✏️ Exercice 9 — Le critère de Kaiser

    Le **critère de Kaiser** (données standardisées) conserve les composantes telles que :

    - **A.** $\lambda_k > 0$
    - **B.** $\lambda_k > 1$
    - **C.** variance cumulée $> 1$
    - **D.** $\lambda_k < 1$
    """)
    return


@app.cell
def _(mo):
    ex9 = mo.ui.radio(options=["A", "B", "C", "D"], label="**Votre réponse :**")
    ex9
    return (ex9,)


@app.cell(hide_code=True)
def _(ex9, verifier_qcm):
    verifier_qcm(ex9.value, "df7e70e5")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✏️ Exercice 10 — Scores vs loadings

    Différence entre **scores** $Z$ et **loadings** $W_k$ :

    - **A.** $Z$ = contribution des variables ; $W_k$ = coordonnées des individus
    - **B.** ils sont identiques
    - **C.** $Z$ = **coordonnées des individus** dans le nouvel espace ; $W_k$ = **contribution des variables** d'origine aux composantes
    - **D.** $Z$ et $W_k$ ont tous deux la forme $(n\times d)$
    """)
    return


@app.cell
def _(mo):
    ex10 = mo.ui.radio(options=["A", "B", "C", "D"], label="**Votre réponse :**")
    ex10
    return (ex10,)


@app.cell(hide_code=True)
def _(ex10, verifier_qcm):
    verifier_qcm(ex10.value, "6b23c0d5")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✏️ Exercice 11 — Reconstruction et variance perdue

    - **A.** $\hat{X}=Z W_k$ ; $1-\text{PVE}(k)$ = variance conservée
    - **B.** $\hat{X}=W_k Z$ ; $1-\text{PVE}(k)$ = nombre de composantes
    - **C.** $\hat{X}=X_c$ ; $1-\text{PVE}(k)=0$
    - **D.** $\hat{X}=Z W_k^\top$ ; $1-\text{PVE}(k)$ = **fraction de variance perdue** (non récupérable)
    """)
    return


@app.cell
def _(mo):
    ex11 = mo.ui.radio(options=["A", "B", "C", "D"], label="**Votre réponse :**")
    ex11
    return (ex11,)


@app.cell(hide_code=True)
def _(ex11, verifier_qcm):
    verifier_qcm(ex11.value, "3f39d5c3")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✏️ Exercice 12 — Pourquoi la SVD ?

    Scikit-learn calcule l'ACP par **SVD de $X_c$** (plutôt que par diagonalisation de
    $\Sigma$) car :

    - **A.** la SVD ne nécessite pas de standardisation
    - **B.** diagonaliser $\Sigma=X_c^\top X_c$ est numériquement instable ; la SVD agit directement sur $X_c$, avec $\lambda_k=\sigma_k^2/(n-1)$
    - **C.** la SVD donne des composantes différentes des vecteurs propres
    - **D.** la SVD est une méthode supervisée
    """)
    return


@app.cell
def _(mo):
    ex12 = mo.ui.radio(options=["A", "B", "C", "D"], label="**Votre réponse :**")
    ex12
    return (ex12,)


@app.cell(hide_code=True)
def _(ex12, verifier_qcm):
    verifier_qcm(ex12.value, "df7e70e5")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 🧪 Partie B — Pratique (Exercices 13 → 16)

    On passe à la mise en œuvre sur le vrai dataset. Les cellules **`Données`** et **`EDA`**
    ci-dessous sont **fournies** : lisez-les, ne les modifiez pas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📂 Données (fournies)
    """)
    return


@app.cell
def donnees(StandardScaler):
    """Chargement du dataset Breast Cancer Wisconsin (Diagnostic) — fournie, ne pas modifier."""
    from sklearn.datasets import load_breast_cancer

    _data = load_breast_cancer()
    X = _data.data
    y = _data.target                     # 0 = maligne (cancéreuse), 1 = bénigne
    feat_names = list(_data.feature_names)
    X_std = StandardScaler().fit_transform(X)

    print(f"Breast Cancer Wisconsin (Diagnostic) : {X.shape[0]} biopsies × {X.shape[1]} variables")
    print(f"  Maligne (0) : {(y == 0).sum():>3}  ({(y == 0).mean():.0%})")
    print(f"  Bénigne (1) : {(y == 1).sum():>3}  ({(y == 1).mean():.0%})")
    print(f"\n  Variables (forme des noyaux) : {feat_names[:4]} …")
    return X, X_std, feat_names, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🔎 EDA express — les variables sont-elles corrélées ? (fournie)

    L'ACP « compresse » d'autant mieux que les variables sont **corrélées** (redondantes).
    Regardons la matrice de corrélation **avant** de lancer l'ACP : cela nous dira à
    l'avance si on peut espérer beaucoup compresser. *(Indice : rayon, périmètre et aire
    d'un noyau sont géométriquement liés…)*
    """)
    return


@app.cell
def eda_correlation(X_std, feat_names, np, plt, sns):
    """Matrice de corrélation — cellule fournie."""
    _corr = np.corrcoef(X_std.T)

    _fig, _ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(_corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.2, ax=_ax,
                xticklabels=feat_names, yticklabels=feat_names)
    _ax.set_title("Corrélations entre les 30 variables — Breast Cancer (Diagnostic)")
    _ax.tick_params(axis="x", rotation=90, labelsize=6)
    _ax.tick_params(axis="y", labelsize=6)

    # Corrélation absolue moyenne hors diagonale
    _mask = ~np.eye(len(feat_names), dtype=bool)
    _mean_abs = np.abs(_corr[_mask]).mean()
    plt.tight_layout()
    print(f"Corrélation absolue moyenne (hors diagonale) : {_mean_abs:.2f}")
    print("→ Question : faible ou élevée ? Qu'est-ce que cela laisse présager pour l'ACP ?")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## ✏️ Exercice 13 — Combien de composantes faut-il garder ?

    Le compte-rendu demandé doit tenir en 2-3 indicateurs. Vérifions ce qui est réaliste.

    **Travail demandé :**

    1. Une ACP **complète** est ajustée sur `X_std` (déjà fait dans la cellule).
    2. **Complétez** la variance cumulée `_var_cum` (indice : `np.cumsum`).
    3. Lisez sur le graphe **combien de composantes** sont nécessaires pour atteindre
       **80 %** puis **90 %** de variance.

    **Questions :** Quelle part de variance capturent les axes principaux PC1 et PC2 ? Combien
    de composantes pour atteindre 80 % ? Pour 90 % ? Comparez au nombre de variables de départ (30) :
    l'ACP compresse-t-elle beaucoup ? Reliez votre réponse à l'EDA précédente.
    """)
    return


@app.cell
def exercice_13(PCA, X_std, mo, np, plt):
    _pca_full = PCA().fit(X_std)
    _var_ratio = _pca_full.explained_variance_ratio_

    # 👉 À COMPLÉTER : variance cumulée à partir de _var_ratio (indice : np.cumsum)
    _var_cum = np.cumsum(_var_ratio)     # ← remplacez None

    if _var_cum is None:
        _out = mo.md("⏳ **Exercice 13** — complétez `_var_cum` pour afficher les graphes.")
    else:
        _c = range(1, len(_var_ratio) + 1)
        _fig, _axes = plt.subplots(1, 2, figsize=(13, 4))

        _axes[0].bar(_c, _var_ratio, color="steelblue", edgecolor="white")
        _axes[0].plot(_c, _var_ratio, "ro-", ms=4)
        _axes[0].set_title("Scree plot"); _axes[0].set_xlabel("Composante")
        _axes[0].set_ylabel("Variance expliquée"); _axes[0].grid(True, alpha=0.3)

        _axes[1].plot(_c, _var_cum, "go-", lw=2)
        for _t, _col in [(0.80, "orange"), (0.90, "red")]:
            _k = int(np.argmax(_var_cum >= _t) + 1)
            _axes[1].axhline(_t, color=_col, ls="--", label=f"{_t:.0%} → k = {_k}")
            _axes[1].axvline(_k, color=_col, ls=":", alpha=0.5)
        _axes[1].set_title("Variance cumulée"); _axes[1].set_xlabel("Nombre de composantes")
        _axes[1].legend(); _axes[1].grid(True, alpha=0.3)
        plt.tight_layout()

        # ── Vérification du code ─────────────────────────────────────────
        _ref = np.cumsum(_var_ratio)
        _ok = (np.ndim(_var_cum) == 1 and len(_var_cum) == len(_ref)
               and np.allclose(_var_cum, _ref))
        _verdict = mo.callout(
            mo.md("✅ **Code correct** — `_var_cum` est bien la variance cumulée."
                  if _ok else
                  "❌ **Pas encore** — `_var_cum` doit valoir `np.cumsum(_var_ratio)` "
                  "(vecteur croissant terminant à 1)."),
            kind="success" if _ok else "danger",
        )

        print(f"PC1 + PC2 → {_var_cum[1]:.1%} de la variance")
        print(f"80 % de variance → {int(np.argmax(_var_cum >= 0.80) + 1)} composantes")
        print(f"90 % de variance → {int(np.argmax(_var_cum >= 0.90) + 1)} composantes")
        _out = mo.vstack([_verdict, _fig])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## ✏️ Exercice 14 — Voir les biopsies en 2D

    La projection sur les **2 premières composantes** permet de visualiser les 569 biopsies
    d'un coup — et ici, contrairement à des données peu corrélées, elle est très parlante.

    **Travail demandé :** calculez `_Z`, la projection de `X_std` sur **2 composantes**
    (`PCA(n_components=2).fit_transform(...)`). Le nuage sera colorié selon le diagnostic.

    **Questions :** les tumeurs malignes et bénignes forment-elles des groupes **séparés** ?
    Sur quel axe la séparation est-elle la plus nette ? Sachant que PC1 + PC2 ne capturent
    « que » ~63 % de la variance, que pensez-vous de la qualité de cette séparation ?
    """)
    return


@app.cell
def exercice_14(PCA, X_std, mo, np, plt, y):
    # 👉 À COMPLÉTER : projection de X_std sur 2 composantes principales
    _Z = PCA(n_components=2).fit_transform(X_std)      # ← remplacez None (indice : PCA(n_components=2).fit_transform)

    if _Z is None:
        _out = mo.md("⏳ **Exercice 14** — calculez `_Z` (projection 2D).")
    else:
        _fig, _ax = plt.subplots(figsize=(7, 6))
        for _lab, _val, _col in [("Maligne", 0, "red"), ("Bénigne", 1, "green")]:
            _m = y == _val
            _ax.scatter(_Z[_m, 0], _Z[_m, 1], c=_col, label=_lab, alpha=0.5, s=25)
        _ax.set_xlabel("PC1"); _ax.set_ylabel("PC2")
        _ax.set_title("Biopsies projetées par l'ACP (2 composantes)")
        _ax.legend(); _ax.grid(True, alpha=0.3)

        # ── Vérification du code ─────────────────────────────────────────
        _Z_ref = PCA(n_components=2).fit_transform(X_std)
        _ok = (np.shape(_Z) == _Z_ref.shape
               and np.allclose(np.abs(_Z), np.abs(_Z_ref), atol=1e-6))
        _verdict = mo.callout(
            mo.md("✅ **Code correct** — `_Z` est bien la projection sur 2 composantes."
                  if _ok else
                  "❌ **Pas encore** — `_Z` doit être `PCA(n_components=2).fit_transform(X_std)` "
                  "(forme 569 × 2)."),
            kind="success" if _ok else "danger",
        )
        _out = mo.vstack([_verdict, _fig])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## ✏️ Exercice 15 — Que « signifient » les composantes ?

    Une composante n'a de valeur que si on sait **ce qu'elle mesure**. On lit cela dans les
    **loadings** : la contribution de chaque variable d'origine à chaque composante.

    **Travail demandé :** calculez `_load`, la matrice des loadings de forme
    (30 variables × 2 composantes). Le loading officiel est le **vecteur propre pondéré
    par $\sqrt{\lambda}$** :
    $\text{Loadings} = \text{vecteurs propres} \times \sqrt{\text{valeurs propres}}$.
    Ainsi mis à l'échelle, un loading vaut la **corrélation** entre la variable et la
    composante (compris entre −1 et +1).

    **Questions :** citez les **3 variables** qui pèsent le plus (en valeur absolue) sur
    **PC1**. Ont-elles un thème commun ? Proposez un **nom** pour l'axe PC1. En reliant à
    l'exercice 14 (les malignes sont d'un côté de PC1), qu'indique une grande valeur de PC1
    sur le plan clinique ?
    """)
    return


@app.cell
def exercice_15(PCA, X_std, feat_names, mo, np, plt, sns):
    _pca = PCA(n_components=2).fit(X_std)

    # 👉 À COMPLÉTER : loadings (30 × 2) = vecteurs propres × sqrt(valeurs propres)
    _load = (_pca.components_ * np.sqrt(_pca.explained_variance_[:, np.newaxis])).T     # ← remplacez None


    if _load is None:
        _out = mo.md("⏳ **Exercice 15** — calculez `_load` (loadings).")
    else:
        _fig, _ax = plt.subplots(figsize=(6, 10))
        sns.heatmap(np.round(_load, 2), annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1, ax=_ax,
                    xticklabels=["PC1", "PC2"], yticklabels=feat_names, linewidths=0.5)
        _ax.set_title("Loadings — contribution des variables")
        _ax.tick_params(axis="y", labelsize=7)

        # ── Vérification du code ─────────────────────────────────────────
        _ref = (_pca.components_ * np.sqrt(_pca.explained_variance_[:, np.newaxis])).T
        _ok = (np.shape(_load) == _ref.shape
               and np.allclose(np.abs(_load), np.abs(_ref), atol=1e-6))
        _verdict = mo.callout(
            mo.md("✅ **Code correct** — `_load` est bien la matrice des loadings (30 × 2)."
                  if _ok else
                  "❌ **Pas encore** — `_load` doit être "
                  "`(_pca.components_ * np.sqrt(_pca.explained_variance_[:, np.newaxis])).T` "
                  "(forme 30 × 2)."),
            kind="success" if _ok else "danger",
        )

        _top = np.argsort(np.abs(_load[:, 0]))[::-1][:3]
        print("Les 3 variables les plus contributives à PC1 :")
        for _j in _top:
            print(f"  {feat_names[_j]:<28} loading = {_load[_j, 0]:+.2f}")
        _out = mo.vstack([_verdict, _fig])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## ✏️ Exercice 16 — L'ACP aide-t-elle à diagnostiquer ?

    Dernière question de la cheffe de service : réduire les dimensions, est-ce que ça
    **dégrade** un modèle de diagnostic automatique ?

    On compare une régression logistique **sur les 30 variables** vs **sur les composantes
    ACP**, en validation croisée (5 plis), pour un `_k` que vous choisissez.

    **Travail demandé :** fixez `_k` = nombre de composantes ACP à tester (essayez `2`, puis
    la valeur trouvée en Ex. 13 pour 90 %, etc.).

    **Questions :** avec seulement **`_k = 2`** composantes (au lieu de 30), perd-on beaucoup
    en accuracy ? Rappel : 63 % des biopsies sont bénignes — que vaut un modèle « tout est
    bénin » ? Vos scores battent-ils largement ce seuil ?
    """)
    return


@app.cell
def exercice_16(PCA, StandardScaler, X, mo, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    # 👉 À COMPLÉTER : nombre de composantes ACP à tester
    _k = 2     # ← remplacez None par un entier (ex. 2, 5, 10…)

    if _k is None:
        _out = mo.md("⏳ **Exercice 16** — fixez `_k` (nombre de composantes).")
    elif not (isinstance(_k, int) and 1 <= _k <= 30):
        _out = mo.callout(
            mo.md("❌ **`_k` invalide** — choisissez un **entier** entre 1 et 30."),
            kind="danger",
        )
    else:
        _pipe_full = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000)),
        ])
        _pipe_pca = Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=_k)),
            ("clf", LogisticRegression(max_iter=5000)),
        ])
        _s_full = cross_val_score(_pipe_full, X, y, cv=5, scoring="accuracy")
        _s_pca = cross_val_score(_pipe_pca, X, y, cv=5, scoring="accuracy")

        _txt = (
            f"Seuil naïf (tout bénin)    : {max((y == 1).mean(), (y == 0).mean()):.3f}\n"
            f"Sans ACP  (30 variables)   : {_s_full.mean():.3f} ± {_s_full.std():.3f}\n"
            f"Avec ACP  ({_k:>2} composantes) : {_s_pca.mean():.3f} ± {_s_pca.std():.3f}"
        )
        print(_txt)
        _verdict = mo.callout(
            mo.md(f"✅ **Paramètre valide** — comparaison lancée avec **k = {_k}** composantes."),
            kind="success",
        )
        _out = mo.vstack([_verdict, mo.md(f"```\n{_txt}\n```")])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 🧠 Questions de synthèse (à rédiger)

    1. Sur ce dataset, l'ACP permet-elle vraiment de résumer 30 variables en 2-3 indicateurs
       **utiles** ? **Pourquoi ?** (reliez à la corrélation moyenne mesurée dans l'EDA).
    2. Pourquoi des mesures comme rayon, périmètre et aire d'un noyau sont-elles très
       corrélées ? En quoi cela favorise-t-il l'ACP ?
    3. L'exercice 14 montre une bonne séparation malignes/bénignes sur PC1. L'ACP a-t-elle
       *utilisé* les étiquettes de diagnostic pour cela ? (ACP = supervisée ou non ?) Que
       faudrait-il utiliser si l'on voulait **maximiser** cette séparation ? *(indice : LDA)*
    4. Comparez avec un dataset dont les variables seraient **peu corrélées** (ex. un scoring
       où chaque variable apporte une info distincte) : l'ACP compresserait-elle autant ?

    > 💡 **La leçon du TP** : l'ACP brille quand les variables sont **redondantes**. Ici, les
    > 30 mesures décrivent quelques traits sous-jacents (surtout la **taille/irrégularité** de
    > la tumeur) → 2-3 composantes suffisent à résumer *et* à séparer. L'ACP n'a pas « appris »
    > le diagnostic : la séparation émerge naturellement de la structure des données.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Réponse Q1 :Oui, l'ACP résume efficacement les 30 variables. La corrélation absolue
    moyenne mesurée dans l'EDA est de 0,39, ce qui montre une redondance notable entre les
    variables (notamment celles liées à la taille du noyau). Cela se confirme avec l'ACP :
    PC1 et PC2 capturent à eux seuls 63,2 % de la variance totale, et il suffit de 5
    composantes pour 80 % et 7 pour 90 %, contre 30 variables de départ.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Réponse Q2:Le rayon, le périmètre et l'aire d'un noyau sont géométriquement liés : un noyau plus grand a mécaniquement un rayon plus grand, ce qui augmente aussi son périmètre et son aire. Ces trois variables mesurent donc, sous des angles différents, la même caractéristique sous-jacente : la taille du noyau. Cette redondance naturelle est justement ce que l'ACP exploite : plutôt que de traiter ces mesures comme trois informations indépendantes, elle les regroupe en un seul axe (une composante), ce qui permet de compresser l'information sans en perdre beaucoup.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Réponse Q3:Non, l'ACP n'utilise pas les étiquettes de diagnostic, c'est une méthode non supervisée : elle se base uniquement sur la variance des données. Si on observe quand même une bonne séparation entre malignes et bénignes en Ex.14, c'est juste parce que ça correspond à une réalité biologique, et ça se retrouve naturellement dans PC1. Pour maximiser volontairement cette séparation en utilisant les étiquettes, il faudrait plutôt utiliser la LDA, qui est supervisée.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Réponse Q4:Non, l'ACP compresserait beaucoup moins bien sur des variables peu corrélées. Ici ça marche parce que les 30 mesures sont redondantes (corrélation moyenne 0,39, beaucoup liées à la taille du noyau). Si chaque variable apportait une info vraiment distincte, il n'y aurait pas de structure commune à exploiter : il faudrait garder presque toutes les composantes pour ne pas perdre d'information, la variance serait répartie uniformément et il n'y aurait pas de PC1 dominant comme ici.
    """)
    return


if __name__ == "__main__":
    app.run()
