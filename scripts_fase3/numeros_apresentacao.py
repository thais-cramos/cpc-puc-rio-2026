"""Números para a apresentação da fase 3:
(1) diferença PUC-Rio − média dos pares do grupo C2 em cada indicador (média das 10 divisões);
(2) Regime dos pares no C2 e no C5;
(3) simulação da meta: quantos horistas converter para chegar à média dos cursos privados da área e o efeito na nota de
    Regime e no CPC (a nota padronizada é uma função linear da nota bruta dentro da área; peso do Regime = 7,5%).
Saídas em resultados_fase3/."""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FOCO = {5710: "Engenharia Civil", 6208: "Engenharia de Produção", 21: "Arquitetura e Urbanismo"}
IND = ["Enade_2023", "IDD_2023", "Mestres_2023", "Doutores_2023", "Regime_Trabalho_2023",
       "Organizacao_Didatica_2023", "Infraestrutura_2023", "Oportunidades_2023"]
NOMES = ["Enade", "IDD", "Mestres", "Doutores", "Regime de trabalho", "Org. didático-pedag.", "Infraestrutura", "Oportunidades"]
IND_C5 = [c for c in IND if c != "Regime_Trabalho_2023"]
PUC = 528
base = pd.read_csv("dados/processados/base_analitica_1907_cursos.csv", encoding="utf-8-sig")
k = pd.read_csv("resultados_notebook_09/k_escolhido_por_area.csv", sep=";", decimal=",").set_index(["Area", "Cenario"])["K_escolhido"]

dif, reg = [], []
for a, nome in FOCO.items():
    d = base[base["Area_2023"] == a].reset_index(drop=True); pos = int(np.flatnonzero(d["Código da IES"].eq(PUC))[0])
    medias = {"C2": [], "C5": []}
    for s in range(42, 52):
        tr, _ = train_test_split(np.arange(len(d)), test_size=0.2, random_state=s)
        for cen, cols, kk in (("C2", IND, k[(a, "C2_8IND")]), ("C5", IND_C5, k[(a, "C5_SEM_REGIME")])):
            g = Pipeline([("e", StandardScaler()), ("k", KMeans(n_clusters=int(kk), random_state=s, n_init=20))]).fit(d.loc[tr, cols]).predict(d[cols])
            par = (g == g[pos]) & (np.arange(len(d)) != pos)
            medias[cen].append(d.loc[par, IND].mean())
    m2, m5 = pd.concat(medias["C2"], axis=1).mean(axis=1), pd.concat(medias["C5"], axis=1).mean(axis=1)
    for c, nm in zip(IND, NOMES):
        dif.append({"Área": nome, "Indicador": nm, "PUC-Rio": d.loc[pos, c], "Pares C2": m2[c], "Diferença": d.loc[pos, c] - m2[c]})
    reg.append({"Área": nome, "PUC-Rio": d.loc[pos, "Regime_Trabalho_2023"], "Pares C2": m2["Regime_Trabalho_2023"], "Pares C5": m5["Regime_Trabalho_2023"],
                "Enade pares C5": m5["Enade_2023"], "Enade PUC": d.loc[pos, "Enade_2023"]})
dif = pd.DataFrame(dif); reg = pd.DataFrame(reg)
dif.round(3).to_csv("resultados_fase3/diferencas_puc_pares_C2.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
reg.round(3).to_csv("resultados_fase3/regime_pares_C2_C5.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")

# simulação da meta
cpc = pd.read_excel("dados/CPC_2023.xlsx", "CPC_2023"); cpc.columns = cpc.columns.str.strip()
cpc = cpc[cpc["Código da Área"].isin(FOCO) & cpc["Modalidade de Ensino"].eq("Educação Presencial")]
pnl = pd.read_csv("resultados_fase3/painel_docentes_regime_2023.csv", sep=";", decimal=",", encoding="utf-8-sig")
sim = []
for a, nome in FOCO.items():
    c = cpc[cpc["Código da Área"] == a].dropna(subset=["Nota Bruta – Regime de Trabalho", "Nota Padronizada - Regime de Trabalho"])
    nao5 = c[(c["Nota Padronizada - Regime de Trabalho"] > 0.01) & (c["Nota Padronizada - Regime de Trabalho"] < 4.99)]
    b1, b0 = np.polyfit(nao5["Nota Bruta – Regime de Trabalho"], nao5["Nota Padronizada - Regime de Trabalho"], 1)
    nota = lambda x: float(np.clip(b0 + b1 * x, 0, 5))
    p = c[c["Código da IES"] == PUC].iloc[0]
    priv = c[~c["Categoria Administrativa"].str.startswith("Pública")]
    alvo = priv["Nota Bruta – Regime de Trabalho"].mean()
    tot = int(pnl[(pnl["Área"] == nome) & (pnl["IES"] == "PUC-Rio")]["Total"].iloc[0])
    hor = int(pnl[(pnl["Área"] == nome) & (pnl["IES"] == "PUC-Rio")]["Horista"].iloc[0])
    converter = int(np.ceil(alvo * tot - (tot - hor)))
    n_atual, n_meta = nota(p["Nota Bruta – Regime de Trabalho"]), nota(alvo)
    sim.append({"Área": nome, "Docentes (painel)": tot, "Horistas (painel)": hor,
                "% não horista atual": 100 * p["Nota Bruta – Regime de Trabalho"], "% não horista meta (média privados)": 100 * alvo,
                "Horistas a converter": converter, "Nota Regime atual": p["Nota Padronizada - Regime de Trabalho"], "Nota Regime atual (reta)": n_atual,
                "Nota Regime na meta": n_meta, "Ganho no CPC contínuo": 0.075 * (n_meta - p["Nota Padronizada - Regime de Trabalho"]),
                "CPC contínuo atual": p["CPC (Contínuo)"], "CPC faixa atual": p["CPC (Faixa)"]})
sim = pd.DataFrame(sim)
sim.round(3).to_csv("resultados_fase3/simulacao_meta_regime.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(dif.pivot(index="Indicador", columns="Área", values="Diferença").round(2).to_string())
print(reg.round(2).to_string(index=False)); print(sim.round(3).T.to_string())
