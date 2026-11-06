"""Funções da Etapa 4 (aprendizado probabilístico) — mesmas do notebook etapa4.ipynb.

Tema: previsão de hospitalização por dengue (Sinan, Portal de Dados Abertos do SUS),
com os mesmos 6 atributos e o mesmo treino/teste da Etapa 1 (ver utils.py).

Conteúdo: MLE e MAP (Bernoulli), Naive Bayes categórico do zero e
Naive Bayes + EM com variável oculta Z (mistura de K componentes por classe).
O notebook define estas funções nas próprias células; este arquivo existe para
o repositório (o enunciado pede em.py) e pode ser importado em outras etapas.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

SEED = 42
ALPHA = 1.0   # Laplace = MAP com prior Beta(2,2) / Dirichlet(2)

# ---------------------------------------------------------------------------
# MLE e MAP (Bernoulli)
# ---------------------------------------------------------------------------
def bernoulli_mle(y):
    """theta_MLE = k / n (fração de sucessos)."""
    y = np.asarray(y)
    return y.sum() / len(y)


def bernoulli_map(y, a=2, b=2):
    """theta_MAP com prior Beta(a, b) = moda da posteriori Beta(k+a, n-k+b).
    Com Beta(2,2): (k + 1) / (n + 2)."""
    y = np.asarray(y)
    k, n = y.sum(), len(y)
    return (k + a - 1) / (n + a + b - 2)


def curvas_beta(k, n, a=2, b=2, pontos=2001):
    """Verossimilhança, prior e posteriori numa grade de theta (normalizadas para área 1)."""
    g = np.linspace(1e-4, 1 - 1e-4, pontos)
    passo = g[1] - g[0]
    log_lik = k * np.log(g) + (n - k) * np.log(1 - g)
    log_pri = (a - 1) * np.log(g) + (b - 1) * np.log(1 - g)

    def normaliza(lp):
        d = np.exp(lp - lp.max())
        return d / (d.sum() * passo)

    return g, normaliza(log_lik), normaliza(log_pri), normaliza(log_lik + log_pri)


# ---------------------------------------------------------------------------
# Codificação dos atributos
# ---------------------------------------------------------------------------
def codificar(tr, te, atributos):
    """Retorna X_treino, X_teste (matrizes de inteiros) e o vocabulário de cada atributo."""
    vocab = {a: sorted(tr[a].unique()) for a in atributos}

    def cod(df):
        return np.column_stack([
            df[a].map({v: i for i, v in enumerate(vocab[a])}).fillna(-1).astype(int).to_numpy()
            for a in atributos])

    return cod(tr), cod(te), vocab


def logsumexp(L):
    """log(sum(exp(L))) por linha, sem estouro numérico."""
    m = L.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(L - m).sum(axis=1, keepdims=True)))[:, 0]


# ---------------------------------------------------------------------------
# Naive Bayes do zero
# ---------------------------------------------------------------------------
def nb_treinar(X, y, cards, alpha=ALPHA):
    """prior[c] = P(C=c);  theta[j][c, v] = P(x_j = v | C = c)."""
    prior = np.array([np.mean(y == c) for c in (0, 1)])
    theta = []
    for j, m in enumerate(cards):
        t = np.zeros((2, m))
        for c in (0, 1):
            cont = np.bincount(X[y == c, j], minlength=m)
            t[c] = (cont + alpha) / (cont.sum() + alpha * m)
        theta.append(t)
    return {"tipo": "nb", "prior": prior, "theta": theta}


def nb_log_conjunta(modelo, X):
    """Matriz (N x 2) com log P(C=c, x) = log P(c) + soma_j log P(x_j | c)."""
    with np.errstate(divide="ignore"):             # alpha = 0 pode gerar log(0) = -inf
        L = np.tile(np.log(modelo["prior"]), (len(X), 1))
        for j, t in enumerate(modelo["theta"]):
            ok = X[:, j] >= 0                      # ignora valor nunca visto no treino
            L[ok] += np.log(t[:, X[ok, j]]).T
    return L


def prob_hosp(L):
    """P(alvo = 1 | x) normalizando log P(c, x) (o alpha da eq. 12.21)."""
    with np.errstate(invalid="ignore"):
        m = L.max(axis=1, keepdims=True)
        e = np.exp(L - m)
        p = e[:, 1] / e.sum(axis=1)
    # se P(x | c) = 0 para as duas classes (só acontece no MLE), não há como decidir: 0,5
    return np.where(np.isfinite(m[:, 0]), p, 0.5)


# ---------------------------------------------------------------------------
# EM com variável oculta Z
# ---------------------------------------------------------------------------
def log_componentes(X, pi, theta):
    """Matriz (N x K) com log[ P(Z=k) * prod_j P(x_j | Z=k) ]."""
    L = np.tile(np.log(pi), (len(X), 1))
    for j, t in enumerate(theta):
        ok = X[:, j] >= 0
        L[ok] += np.log(t[:, X[ok, j]]).T
    return L


def em_mistura(X, cards, K, alpha=ALPHA, max_iter=300, tol=1e-5, seed=SEED):
    """EM para uma mistura de K componentes categóricos (exemplos de UMA classe)."""
    rng = np.random.default_rng(seed)
    N = len(X)
    pi = np.full(K, 1.0 / K)
    theta = [rng.dirichlet(np.ones(m), size=K) for m in cards]   # inicialização aleatória
    hist_ll, hist_obj = [], []

    for it in range(max_iter):
        # ---- E-step: responsabilidades r[n, k] = P(Z = k | x_n)
        L = log_componentes(X, pi, theta)
        log_px = logsumexp(L)                    # log P(x_n)
        r = np.exp(L - log_px[:, None])          # normaliza cada linha

        hist_ll.append(log_px.sum())
        hist_obj.append(log_px.sum() + alpha * sum(np.log(t).sum() for t in theta))
        if it > 0 and abs(hist_obj[-1] - hist_obj[-2]) / N < tol:
            break

        # ---- M-step: reestima P(Z) e P(x_j | Z) com contagens esperadas
        Nk = r.sum(axis=0)
        pi = Nk / N
        theta = []
        for j, m in enumerate(cards):
            cont = np.stack([r[X[:, j] == v].sum(axis=0) for v in range(m)], axis=1)  # K x m
            theta.append((cont + alpha) / (cont.sum(axis=1, keepdims=True) + alpha * m))

    return {"pi": pi, "theta": theta, "hist_ll": hist_ll, "hist_obj": hist_obj,
            "iteracoes": len(hist_ll)}


def nbem_treinar(X, y, cards, K, alpha=ALPHA, n_init=5, **kw):
    """Um EM por classe (melhor de n_init inicializações). prior[c] = P(C=c)."""
    prior = np.array([np.mean(y == c) for c in (0, 1)])
    comps = []
    for c in (0, 1):
        rodadas = [em_mistura(X[y == c], cards, K, alpha, seed=SEED + s, **kw) for s in range(n_init)]
        comps.append(max(rodadas, key=lambda m: m["hist_obj"][-1]))
    return {"tipo": "em", "K": K, "prior": prior, "comps": comps}


def nbem_log_conjunta(modelo, X):
    """Matriz (N x 2): log P(C=c, x) = log P(c) + log soma_k P(Z=k|c) prod_j P(x_j | k, c)."""
    return np.column_stack([
        np.log(modelo["prior"][c]) + logsumexp(log_componentes(X, comp["pi"], comp["theta"]))
        for c, comp in enumerate(modelo["comps"])])


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------
def treinar(X, y, cards, K):
    return nb_treinar(X, y, cards) if K == 1 else nbem_treinar(X, y, cards, K)


def log_conjunta(modelo, X):
    return nb_log_conjunta(modelo, X) if modelo["tipo"] == "nb" else nbem_log_conjunta(modelo, X)


def metricas(y, p):
    return {"acuracia": np.mean((p >= 0.5).astype(int) == y),
            "auc": roc_auc_score(y, p),
            "log_loss": log_loss(y, np.clip(p, 1e-12, 1 - 1e-12))}


def cv_modelo(dados, atributos, K, k=5, seed=SEED):
    """Métricas por fold (validação cruzada estratificada) do NB (K=1) ou NB + EM (K>=2)."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    dados = dados.reset_index(drop=True)
    y = dados["alvo"].to_numpy()
    res = []
    for tr, va in skf.split(dados, y):
        Xa, Xb, vocab = codificar(dados.iloc[tr], dados.iloc[va], atributos)
        modelo = treinar(Xa, y[tr], [len(vocab[a]) for a in atributos], K)
        res.append(metricas(y[va], prob_hosp(log_conjunta(modelo, Xb))))
    return res


def resumo_cv(res):
    return {m: np.mean([r[m] for r in res]) for m in ("acuracia", "auc", "log_loss")}
