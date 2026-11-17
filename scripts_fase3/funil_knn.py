"""Funil do critério de vizinhança do k-NN (divisão de semente 42, a do modelo final): quantos cursos sobram a cada filtro."""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FOCO = {5710: "Engenharia Civil", 6208: "Engenharia de Produção", 21: "Arquitetura e Urbanismo"}
IND = ["Enade_2023", "IDD_2023", "Mestres_2023", "Doutores_2023", "Regime_Trabalho_2023",
       "Organizacao_Didatica_2023", "Infraestrutura_2023", "Oportunidades_2023"]
base = pd.read_csv("dados/processados/base_analitica_1907_cursos.csv", encoding="utf-8-sig")
cpc = pd.read_excel("dados/CPC_2023.xlsx", "CPC_2023"); cpc.columns = cpc.columns.str.strip()
priv = set(cpc.loc[~cpc["Categoria Administrativa"].str.startswith("Pública"), "Código do Curso"])
k = pd.read_csv("resultados_notebook_09/k_escolhido_por_area.csv", sep=";", decimal=",").query("Cenario == 'C2_8IND'").set_index("Area")["K_escolhido"]
ref = pd.read_csv("resultados_fase3/knn_referencias.csv", sep=";", decimal=",", encoding="utf-8-sig")
linhas = []
for a, nome in FOCO.items():
    d = base[base["Area_2023"] == a].reset_index(drop=True); pos = int(np.flatnonzero(d["Código da IES"].eq(528))[0])
    tr, _ = train_test_split(np.arange(len(d)), test_size=0.2, random_state=42)
    g = Pipeline([("e", StandardScaler()), ("k", KMeans(n_clusters=int(k[a]), random_state=42, n_init=20))]).fit(d.loc[tr, IND]).predict(d[IND])
    grupo = d[(g == g[pos]) & (np.arange(len(d)) != pos)]
    privs = grupo[grupo["Código do Curso"].isin(priv)]
    corte = max(d.loc[pos, "Regime_Trabalho_2023"], d["Regime_Trabalho_2023"].mean())
    cand = privs[privs["Regime_Trabalho_2023"] > corte]
    r = ref[(ref["Área"] == nome) & (ref["Papel"] == "Referência")]
    linhas.append({"Área": nome, "Cursos na área": len(d), "No grupo da PUC-Rio": len(grupo), "Privados do grupo": len(privs),
                   "Regime acima da média da área": len(cand), "5 mais próximas": len(r), "Estáveis (≥ 50% das divisões)": int((r["% das 10 divisões"] >= 50).sum())})
f = pd.DataFrame(linhas); f.to_csv("resultados_fase3/funil_knn.csv", sep=";", index=False, encoding="utf-8-sig")
print(f.to_string(index=False))
