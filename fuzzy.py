"""Funções da Etapa 6 (lógica fuzzy) — as mesmas do notebook etapa6.ipynb.
 
Tema: previsão de hospitalização por dengue (Sinan, Portal de Dados Abertos do SUS).
 
Sistema de inferência fuzzy do tipo Mamdani, implementado do zero com NumPy:
  entradas: p_rede (probabilidade da rede Bayesiana da Etapa 5, de 0 a 1)
            sinais (número de sinais de alarme: leucopenia + vômito, de 0 a 2)
  saída:    risco de hospitalização (0 a 100); decisão "hospitalizar" se risco >= 50.
O notebook define estas funções nas próprias células; este arquivo existe para
o repositório (o enunciado pede fuzzy.py) e pode ser importado em outros scripts.
"""
import numpy as np
 
U_P = np.linspace(0, 1, 101)       # universo de p_rede
U_S = np.linspace(0, 2, 201)       # universo de sinais
U_R = np.linspace(0, 100, 1001)    # universo de risco
 
 
# ---------------------------------------------------------------------------
# Funções de pertinência, operadores e defuzzificação
# ---------------------------------------------------------------------------
def triangular(x, a, b, c):
    """Pertinência triangular: 0 em a, 1 em b, 0 em c."""
    x = np.asarray(x, dtype=float)
    sobe = (x - a) / (b - a) if b > a else np.ones_like(x)
    desce = (c - x) / (c - b) if c > b else np.ones_like(x)
    return np.clip(np.minimum(sobe, desce), 0.0, 1.0)
 
 
def trapezoidal(x, a, b, c, d):
    """Pertinência trapezoidal; com a = b (ou c = d) vira um 'ombro' que vale 1 até a borda."""
    x = np.asarray(x, dtype=float)
    sobe = np.ones_like(x) if a == b else np.clip((x - a) / (b - a), 0.0, 1.0)
    desce = np.ones_like(x) if c == d else np.clip((d - x) / (d - c), 0.0, 1.0)
    return np.where((x < a) | (x > d), 0.0, np.minimum(sobe, desce))
 
 
def E(*graus):
    return min(graus)        # t-norma do mínimo
 
 
def OU(*graus):
    return max(graus)        # s-norma do máximo
 
 
def centroide(universo, mu):
    """Centro de massa do conjunto agregado (integral pela regra dos trapézios)."""
    def area(f):
        return np.sum((universo[1:] - universo[:-1]) * (f[1:] + f[:-1]) / 2)
    return float(area(universo * mu) / area(mu))
 
 
# ---------------------------------------------------------------------------
# Variáveis linguísticas
# ---------------------------------------------------------------------------
def fuzz_p(p):
    return {"baixa": trapezoidal(p, 0.0, 0.0, 0.2, 0.5),
            "media": triangular(p, 0.2, 0.5, 0.8),
            "alta":  trapezoidal(p, 0.5, 0.8, 1.0, 1.0)}
 
 
def fuzz_s(s):
    return {"nenhum": trapezoidal(s, 0.0, 0.0, 0.5, 1.0),
            "um":     triangular(s, 0.5, 1.0, 1.5),
            "dois":   trapezoidal(s, 1.0, 1.5, 2.0, 2.0)}
 
 
SAIDA = {"baixo":    trapezoidal(U_R, 0, 0, 25, 50),
         "moderado": triangular(U_R, 25, 50, 75),
         "alto":     trapezoidal(U_R, 50, 75, 100, 100)}
 
 
# ---------------------------------------------------------------------------
# Regras e inferência de Mamdani
# ---------------------------------------------------------------------------
REGRAS = [
    ("Se p_rede é alta OU sinais é dois, então risco é alto", lambda P, S: OU(P["alta"], S["dois"]), "alto"),
    ("Se p_rede é média, então risco é moderado",             lambda P, S: P["media"],              "moderado"),
    ("Se p_rede é baixa, então risco é baixo",                lambda P, S: P["baixa"],              "baixo"),
    ("Se p_rede é média E sinais é um, então risco é alto",   lambda P, S: E(P["media"], S["um"]),  "alto"),
]
 
 
def inferir(p, s, detalhe=False):
    """Ativação de cada regra, corte pelo mínimo, agregação pelo máximo, centróide."""
    P = {k: float(v) for k, v in fuzz_p(p).items()}
    S = {k: float(v) for k, v in fuzz_s(s).items()}
    agregado = np.zeros_like(U_R)
    ativacoes = []
    for texto, antecedente, consequente in REGRAS:
        forca = antecedente(P, S)
        ativacoes.append((texto, forca))
        agregado = np.maximum(agregado, np.minimum(forca, SAIDA[consequente]))
    r = centroide(U_R, agregado)
    return (r, P, S, ativacoes, agregado) if detalhe else r
 
 
def decidir(p, s, limiar=50.0):
    """1 = hospitalizar, 0 = não hospitalizar."""
    return int(inferir(p, s) >= limiar)
 
