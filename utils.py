"""Funções auxiliares do trabalho de Fundamentos de IA (Etapas 1 e 2).

Tema: previsão de hospitalização por dengue com os microdados do Sinan
(Portal de Dados Abertos do SUS).

Etapa 1: árvore de decisão (ID3 do zero, entropia, ganho de informação).
Etapa 2: regressão linear (fórmula fechada e gradiente descendente), Ridge
         e regressão logística, todas implementadas do zero.
"""
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42

# Atributos finais da árvore (Etapa 1) e da regressão logística (Etapa 2)
ATRIBUTOS = ["regiao", "LEUCOPENIA", "faixa_etaria", "VOMITO", "CEFALEIA", "HIPERTENSA"]

# Atributos candidatos avaliados na seleção (Etapa 1)
CANDIDATOS = ["regiao", "SG_UF_NOT", "faixa_etaria", "CS_SEXO", "VOMITO",
              "PETEQUIA_N", "DIABETES", "HIPERTENSA", "CEFALEIA", "LEUCOPENIA",
              "MIALGIA", "DOR_RETRO", "ARTRALGIA", "LACO", "EXANTEMA",
              "NAUSEA", "DOR_COSTAS"]

# Colunas lidas do CSV (as demais não são usadas, o que poupa memória)
COLUNAS_ETAPA1 = ["HOSPITALIZ", "NU_IDADE_N", "SG_UF_NOT", "CS_SEXO", "VOMITO",
                  "PETEQUIA_N", "DIABETES", "HIPERTENSA", "CEFALEIA", "LEUCOPENIA",
                  "MIALGIA", "DOR_RETRO", "ARTRALGIA", "LACO", "EXANTEMA",
                  "NAUSEA", "DOR_COSTAS"]
COLUNAS_ETAPA2 = ["HOSPITALIZ", "NU_IDADE_N", "ID_MUNICIP", "SG_UF_NOT",
                  "VOMITO", "LEUCOPENIA", "CEFALEIA", "HIPERTENSA"]

# Atributos da regressão por município (Etapa 2)
FEATS_MUN = ["p_idoso", "p_crianca", "p_vomito", "p_leucopenia",
             "p_cefaleia", "p_hipertensao", "log_casos"]


# ---------------------------------------------------------------------------
# Pré-processamento
# ---------------------------------------------------------------------------
def faixa_etaria(x):
    """NU_IDADE_N: 1o dígito = unidade (4 = anos), demais = valor. Ex.: 4048 = 48 anos."""
    if pd.isna(x):
        return "ignorado"
    x = int(x)
    unidade, valor = x // 1000, x % 1000
    anos = valor if unidade == 4 else 0     # horas, dias e meses = menos de 1 ano
    if anos < 12:
        return "crianca"
    elif anos < 60:
        return "adulto"
    return "idoso"


def regiao(x):
    """SG_UF_NOT: código IBGE do estado; o 1o dígito é a região."""
    if pd.isna(x):
        return "ignorado"
    d = str(int(float(x)))[0]
    return {"1": "norte", "2": "nordeste", "3": "sudeste",
            "4": "sul", "5": "centro-oeste"}.get(d, "ignorado")


def preparar(df, atributos, seed=SEED):
    """Atributos como texto (nulo -> 'ignorado') e undersampling 50/50."""
    d = df[atributos + ["alvo"]].copy()
    for a in atributos:
        d[a] = d[a].fillna("ignorado").astype(str)
    pos = d[d["alvo"] == 1]
    neg = d[d["alvo"] == 0].sample(n=len(pos), random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


def dummies(tr, te, attrs, drop_first=False):
    """Atributos categóricos -> colunas 0/1, com as mesmas colunas em treino e teste."""
    todos = pd.get_dummies(pd.concat([tr[attrs], te[attrs]], keys=["tr", "te"]),
                           drop_first=drop_first).astype(float)
    return todos.loc["tr"], todos.loc["te"]


def treino_teste_etapa1(caminho_csv, seed=SEED):
    """Reconstrói o treino/teste da Etapa 1 (6 atributos finais) direto do CSV bruto."""
    dd = pd.read_csv(caminho_csv, sep=",", encoding="latin-1",
                     usecols=["HOSPITALIZ", "NU_IDADE_N", "SG_UF_NOT", "VOMITO",
                              "LEUCOPENIA", "CEFALEIA", "HIPERTENSA"],
                     low_memory=False)
    n_total = len(dd)
    dd = dd[dd["HOSPITALIZ"].isin([1, 2])].copy()
    dd["alvo"] = (dd["HOSPITALIZ"] == 1).astype(int)
    dd["faixa_etaria"] = dd["NU_IDADE_N"].apply(faixa_etaria)
    dd["regiao"] = dd["SG_UF_NOT"].apply(regiao)
    bal = preparar(dd, ATRIBUTOS, seed)
    tr, te = train_test_split(bal, test_size=0.2, random_state=seed,
                              stratify=bal["alvo"])
    info = {"registros": n_total, "validos": len(dd), "balanceado": len(bal)}
    return tr.reset_index(drop=True), te.reset_index(drop=True), info


# ---------------------------------------------------------------------------
# Etapa 1: árvore de decisão (ID3)
# ---------------------------------------------------------------------------
def entropia(y):
    y = list(y)
    if len(y) == 0:
        return 0.0
    p = np.array(list(Counter(y).values())) / len(y)
    return max(0.0, float(-np.sum(p * np.log2(p))))


def ganho(df, attr, alvo="alvo"):
    h_pai = entropia(df[alvo])
    h_filhos = sum(len(sub) / len(df) * entropia(sub[alvo])
                   for _, sub in df.groupby(attr))
    return h_pai - h_filhos


def arvore(df, atributos, alvo="alvo", max_prof=None, prof=0):
    """ID3 com entropia. max_prof=None -> sem poda."""
    maj = int(df[alvo].mode()[0])
    if df[alvo].nunique() == 1:
        return int(df[alvo].iloc[0])
    if not atributos or (max_prof is not None and prof >= max_prof):
        return maj
    A = max(atributos, key=lambda a: ganho(df, a, alvo))
    return {
        "atributo": A,
        "maj": maj,
        "filhos": {v: arvore(sub, [a for a in atributos if a != A],
                             alvo, max_prof, prof + 1)
                   for v, sub in df.groupby(A)},
    }


def prever(no, linha):
    while isinstance(no, dict):
        v = linha[no["atributo"]]
        if v not in no["filhos"]:      # valor nunca visto neste ramo
            return no["maj"]
        no = no["filhos"][v]
    return no


def acuracia(no, df, alvo="alvo"):
    linhas = df.to_dict("records")
    return sum(prever(no, l) == l[alvo] for l in linhas) / len(linhas)


def cv_id3(dados, atributos, max_prof=None, k=5, seed=SEED):
    """Acurácia por fold (validação cruzada estratificada) do ID3."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    dados = dados.reset_index(drop=True)
    accs = []
    for tr, va in skf.split(dados, dados["alvo"]):
        m = arvore(dados.iloc[tr], atributos, max_prof=max_prof)
        accs.append(acuracia(m, dados.iloc[va]))
    return accs


def relatorio_niveis(df, atributos, niveis=3, rotulo="raiz", nivel=1):
    """Imprime n, entropia e ganhos de cada nó até o nível indicado."""
    n = len(df)
    c0, c1 = int((df["alvo"] == 0).sum()), int((df["alvo"] == 1).sum())
    H = entropia(df["alvo"])
    ind = "    " * (nivel - 1)
    print(f"{ind}[nivel {nivel}] {rotulo} | n={n} (nao={c0}, sim={c1}) | H={H:.4f}")
    if H == 0 or not atributos:
        print(f"{ind}   (folha)")
        return
    gs = sorted(((a, ganho(df, a)) for a in atributos), key=lambda t: -t[1])
    print(f"{ind}   ganhos: " + " | ".join(f"{a}={g:.5f}" for a, g in gs))
    melhor = gs[0][0]
    print(f"{ind}   -> divide por: {melhor}")
    if nivel < niveis:
        resto = [a for a in atributos if a != melhor]
        for v, sub in df.groupby(melhor):
            relatorio_niveis(sub, resto, niveis, f"{melhor}={v}", nivel + 1)


# ---------------------------------------------------------------------------
# Etapa 2: tabela por município e regressões
# ---------------------------------------------------------------------------
def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def tabela_municipios(caminho_csv, min_casos=100):
    """Uma linha por município: taxa de hospitalização (alvo) e proporções (atributos)."""
    d = pd.read_csv(caminho_csv, sep=",", encoding="latin-1",
                    usecols=COLUNAS_ETAPA2, low_memory=False)
    print("registros no arquivo:", len(d))
    d = d[d["HOSPITALIZ"].isin([1, 2])].copy()
    print("com desfecho válido (HOSPITALIZ = 1 ou 2):", len(d))

    d["hosp"] = (d["HOSPITALIZ"] == 1).astype(int)
    d["faixa"] = d["NU_IDADE_N"].apply(faixa_etaria)
    d["idoso"] = (d["faixa"] == "idoso").astype(int)
    d["crianca"] = (d["faixa"] == "crianca").astype(int)
    for c in ["VOMITO", "LEUCOPENIA", "CEFALEIA", "HIPERTENSA"]:
        d[c] = (d[c] == 1).astype(int)       # 1 = sim; nulo e 2 contam como "não"

    mun = d.groupby("ID_MUNICIP").agg(
        n_casos=("hosp", "size"),
        taxa_hosp=("hosp", "mean"),
        p_idoso=("idoso", "mean"),
        p_crianca=("crianca", "mean"),
        p_vomito=("VOMITO", "mean"),
        p_leucopenia=("LEUCOPENIA", "mean"),
        p_cefaleia=("CEFALEIA", "mean"),
        p_hipertensao=("HIPERTENSA", "mean"),
        uf=("SG_UF_NOT", "first"),
    ).reset_index()
    print("municípios (todos):", len(mun))
    return mun[mun["n_casos"] >= min_casos].copy()


def gd_linear(X, y, alpha=0.05, n_iter=2000):
    """Regressão linear por gradiente descendente (Listing 2, versão vetorizada).

    X já deve conter a coluna de 1s (intercepto) e atributos padronizados.
    """
    N, d = X.shape
    w = np.zeros(d)
    hist = []
    for _ in range(n_iter):
        erro = y - X @ w              # y[j] - produto_interno(w, X[j])
        grad = X.T @ erro             # soma de erro * X[j]
        w = w + alpha * grad / N
        hist.append(np.mean(erro ** 2))
    return w, hist


def ridge_gd(X, y, lam, n_iter=5000):
    """Ridge (L2) por gradiente descendente; o intercepto não é penalizado."""
    N, d = X.shape
    L = np.linalg.eigvalsh(X.T @ X / N).max() + lam
    alpha = 1.0 / L                      # passo seguro para qualquer lambda
    w = np.zeros(d)
    pen = np.ones(d)
    pen[0] = 0
    for _ in range(n_iter):
        erro = y - X @ w
        w = w + alpha * (X.T @ erro / N - lam * pen * w)
    return w


def ridge_fechada(X, y, lam):
    """Solução fechada do Ridge (para conferência e para a curva de coeficientes)."""
    N, d = X.shape
    P = np.eye(d)
    P[0, 0] = 0
    return np.linalg.solve(X.T @ X / N + lam * P, X.T @ y / N)


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def log_loss(y, p):
    eps = 1e-12
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def gd_logistica(X, y, n_iter=3000):
    """Regressão logística por gradiente descendente (entropia cruzada)."""
    N, d = X.shape
    L = 0.25 * np.linalg.eigvalsh(X.T @ X / N).max()
    alpha = 1.0 / L
    w = np.zeros(d)
    hist = []
    for _ in range(n_iter):
        p = sigmoide(X @ w)
        w = w + alpha * X.T @ (y - p) / N
        hist.append(log_loss(y, p))
    return w, hist


def acuracia_log(w, X, y):
    return np.mean((sigmoide(X @ w) >= 0.5).astype(int) == y)
