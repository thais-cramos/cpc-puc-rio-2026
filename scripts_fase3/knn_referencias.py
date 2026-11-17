"""Fase 3 — instituições de referência para a PUC-Rio com o novo critério de vizinhança (k-NN).

Critério (modelo conceitual, classe ConfiguracaoKNN):
  1. busca só entre cursos da mesma área, do mesmo grupo do C2 que a PUC-Rio e de instituições privadas;
  2. a distância usa apenas atributos que a PUC-Rio não controla diretamente: perfil dos estudantes (Enade e IDD),
     porte do corpo docente e tipo de instituição — sem titulação nem regime, que são os indicadores de gestão;
  3. só entra como referência quem tem nota de Regime de trabalho acima da média da área (patamar a alcançar);
  4. estabilidade: o grupo C2 é refeito nas 10 divisões (sementes 42 a 51) e conta-se em quantas a instituição volta.
Saída: resultados_fase3/knn_referencias.csv e knn_estabilidade.csv.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

FOCO = {5710: "Engenharia Civil", 6208: "Engenharia de Produção", 21: "Arquitetura e Urbanismo"}
IND = ["Enade_2023", "IDD_2023", "Mestres_2023", "Doutores_2023", "Regime_Trabalho_2023",
       "Organizacao_Didatica_2023", "Infraestrutura_2023", "Oportunidades_2023"]
PUC, SEMENTES, N_REF = 528, list(range(42, 52)), 5
ATRIB_DIST = ["Conceito Enade (Contínuo)", "Nota Padronizada - IDD", "IES docentes (log)",
              "Universidade", "Comunitária", "Sem fins lucrativos"]
SAIDA = Path("resultados_fase3"); SAIDA.mkdir(exist_ok=True)

cpc = pd.read_excel("dados/CPC_2023.xlsx", "CPC_2023"); cpc.columns = cpc.columns.str.strip()
cpc = cpc[cpc["Código da Área"].isin(FOCO) & cpc["Modalidade de Ensino"].eq("Educação Presencial")].copy()
cpc["Privada"] = ~cpc["Categoria Administrativa"].str.startswith("Pública")
cpc["% não horista (curso)"] = 100 * cpc["Nota Bruta – Regime de Trabalho"]
ies = pd.read_csv("dados/MICRODADOS_ED_SUP_IES_2023.CSV", sep=";", encoding="latin1", low_memory=False)
ies = ies[ies["QT_DOC_EXE"] > 0].copy()
ies = ies.assign(**{"IES docentes": ies["QT_DOC_EXE"], "IES docentes (log)": np.log10(ies["QT_DOC_EXE"]),
                    "Universidade": ies["TP_ORGANIZACAO_ACADEMICA"].eq(1).astype(int),
                    "Comunitária": ies["IN_COMUNITARIA"].fillna(0).astype(int),
                    "Sem fins lucrativos": ies["TP_CATEGORIA_ADMINISTRATIVA"].eq(5).astype(int)})
cursos = cpc.merge(ies.set_index("CO_IES")[["SG_IES", "IES docentes", "IES docentes (log)", "Universidade", "Comunitária",
                                            "Sem fins lucrativos"]], left_on="Código da IES", right_index=True, how="inner")
base = pd.read_csv("dados/processados/base_analitica_1907_cursos.csv", encoding="utf-8-sig")
k_c2 = pd.read_csv("resultados_notebook_09/k_escolhido_por_area.csv", sep=";", decimal=",").query("Cenario == 'C2_8IND'").set_index("Area")["K_escolhido"]

def pares_c2(a, semente):
    d = base[base["Area_2023"] == a].reset_index(drop=True)
    tr, _ = train_test_split(np.arange(len(d)), test_size=0.2, random_state=semente)
    g = Pipeline([("e", StandardScaler()), ("k", KMeans(n_clusters=int(k_c2[a]), random_state=42, n_init=20))]).fit(d.loc[tr, IND]).predict(d[IND])
    pos = int(np.flatnonzero(d["Código da IES"].eq(PUC))[0])
    return set(d.loc[(g == g[pos]) & (np.arange(len(d)) != pos), "Código do Curso"])

def referencias(a, pares):
    na = cursos["Código da Área"].eq(a)
    puc = cursos[na & cursos["Código da IES"].eq(PUC)].iloc[0]
    cand = cursos[na & cursos["Código do Curso"].isin(pares) & cursos["Privada"] & cursos["Código da IES"].ne(PUC)].dropna(subset=ATRIB_DIST)
    corte = max(puc["Nota Padronizada - Regime de Trabalho"], base.loc[base["Area_2023"] == a, "Regime_Trabalho_2023"].mean())
    cand = cand[cand["Nota Padronizada - Regime de Trabalho"] > corte]
    # k-NN (scikit-learn NearestNeighbors, distância euclidiana sobre atributos padronizados).
    # Pede todos os vizinhos ordenados e fica com os N_REF mais próximos de instituições diferentes
    # (uma instituição pode ter mais de um curso candidato; vale o curso mais próximo).
    knn = Pipeline([("e", StandardScaler()), ("knn", NearestNeighbors(metric="euclidean"))]).fit(cand[ATRIB_DIST])
    Xp = knn.named_steps["e"].transform(puc[ATRIB_DIST].to_frame().T.astype(float))
    dist, idx = knn.named_steps["knn"].kneighbors(Xp, n_neighbors=len(cand))
    viz = cand.iloc[idx[0]].assign(Distância=dist[0])
    return puc, viz.drop_duplicates("Código da IES").head(N_REF), len(cand)

linhas, estab, por_div = [], [], []
for a, nome in FOCO.items():
    cont = {}
    for s in SEMENTES:
        pares = pares_c2(a, s); _, rr, nc = referencias(a, pares)
        por_div.append({"Área": nome, "Semente": s, "Pares no grupo C2": len(pares), "Candidatas": nc,
                        "Referências": ", ".join(rr["SG_IES"].fillna("").astype(str)), "Distância média": rr["Distância"].mean()})
        for c in rr["Código da IES"]:
            cont[c] = cont.get(c, 0) + 1
    puc, refs, n_cand = referencias(a, pares_c2(a, 42))
    cols = ["Sigla da IES", "Nome da IES", "Sigla da UF", "Conceito Enade (Contínuo)", "Nota Padronizada - IDD", "IES docentes",
            "Nota Padronizada - Regime de Trabalho", "% não horista (curso)", "Nota Padronizada - Doutores", "Nota Padronizada - Mestres"]
    linhas.append({"Área": nome, "Papel": "PUC-Rio", "Distância": 0.0, "% das 10 divisões": np.nan, **puc[cols].to_dict(),
                   "Código da IES": PUC, "Candidatas": n_cand})
    for _, r in refs.iterrows():
        linhas.append({"Área": nome, "Papel": "Referência", "Distância": r["Distância"], "% das 10 divisões": 10 * cont.get(r["Código da IES"], 0),
                       **r[cols].to_dict(), "Código da IES": r["Código da IES"], "Candidatas": n_cand})
    estab.append({"Área": nome, "Candidatas (privadas do grupo, Regime acima da média da área)": n_cand,
                  "Estabilidade média das 5 referências (%)": np.mean([10 * cont.get(c, 0) for c in refs["Código da IES"]])})
ref = pd.DataFrame(linhas); est = pd.DataFrame(estab)
ref.round(3).to_csv(SAIDA / "knn_referencias.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
est.round(1).to_csv(SAIDA / "knn_estabilidade.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(ref[["Área", "Papel", "Sigla da IES", "Sigla da UF", "Distância", "% das 10 divisões", "Conceito Enade (Contínuo)",
           "Nota Padronizada - IDD", "IES docentes", "Nota Padronizada - Regime de Trabalho", "% não horista (curso)"]].round(2).to_string(index=False))
print(est.round(1).to_string(index=False))

# registro no MLflow: uma execução por área com o critério, e uma filha por divisão (grupo refeito no treino daquela semente)
import sys, time
sys.path.insert(0, str(Path(__file__).parent))
from registro_mlflow import iniciar
mlflow = iniciar()
pdv = pd.DataFrame(por_div); pdv.to_csv(SAIDA / "knn_por_divisao.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
for (a, nome), e in zip(FOCO.items(), estab):
    r = ref[(ref["Área"] == nome) & (ref["Papel"] == "Referência")]
    with mlflow.start_run(run_name=f"k-NN referências · {nome}", tags={"modelo": "kNN (distância euclidiana padronizada)", "etapa": "diagnostico",
                                                                       "data": time.strftime("%Y-%m-%d %H:%M")}):
        mlflow.log_params({"area": nome, "area_codigo": a, "k_vizinhos": N_REF, "algoritmo": "sklearn NearestNeighbors (euclidiana, atributos padronizados)",
                           "atributos_distancia": ", ".join(ATRIB_DIST),
                           "filtros": "mesma area; mesmo grupo C2 da PUC-Rio; privadas; Regime acima da media da area",
                           "k_grupos_C2": int(k_c2[a]), "treino": "80%", "sementes": "42-51",
                           "referencias_semente_42": ", ".join(r["Sigla da IES"].fillna("").astype(str))})
        mlflow.log_metrics({"candidatas": e["Candidatas (privadas do grupo, Regime acima da média da área)"],
                            "estabilidade_media_pct": e["Estabilidade média das 5 referências (%)"],
                            "referencias_estaveis": int((r["% das 10 divisões"] >= 50).sum()), "distancia_media": r["Distância"].mean()})
        for _, d in pdv[pdv["Área"] == nome].iterrows():
            with mlflow.start_run(run_name=f"{nome} · semente {d['Semente']}", nested=True, tags={"nivel": "divisao"}):
                mlflow.log_params({"semente": int(d["Semente"]), "referencias": d["Referências"] or "-"})
                mlflow.log_metrics({"pares_grupo_c2": d["Pares no grupo C2"], "candidatas": d["Candidatas"], "distancia_media": d["Distância média"]})
        for arq in ("knn_referencias.csv", "knn_estabilidade.csv", "knn_por_divisao.csv"):
            mlflow.log_artifact(str(SAIDA / arq))
print("MLflow ok")

