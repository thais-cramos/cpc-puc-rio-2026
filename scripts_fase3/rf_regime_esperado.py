"""Fase 3 — Regime de trabalho esperado pelo perfil da instituição (Random Forest) e resíduo da PUC-Rio.

Pergunta: quanto do % de docentes não horistas de um curso é explicado pelo perfil docente da instituição (Censo 2023)
e pela área, e quanto a PUC-Rio se afasta do que esse perfil prevê?
Alvo: % de docentes não horistas do curso (nota bruta do Regime, CPC 2023). Base: cursos presenciais de instituições
privadas das 8 áreas, sem a PUC-Rio. Atributos: perfil da IES no Censo e a área — nenhum atributo do próprio curso
(evita vazamento do alvo). Comparação com dois modelos de base: prever a média do treino e regressão linear.
Protocolo: 10 divisões (sementes 42 a 51) com teste de 20% das instituições (cursos da mesma IES nunca dos dois lados);
parâmetros do RF escolhidos por validação cruzada de 5 partes agrupada por IES, no treino.
Saída: resultados_fase3/rf_qualidade.csv, rf_puc.csv e a figura do artigo.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score

AREAS_8 = [5710, 6208, 21, 5902, 5806, 6307, 6008, 5814]
FOCO = {5710: "Engenharia Civil", 6208: "Engenharia de Produção", 21: "Arquitetura e Urbanismo"}
PUC, SEMENTES, ALVO = 528, list(range(42, 52)), "% não horista (curso)"
SAIDA = Path("resultados_fase3"); SAIDA.mkdir(exist_ok=True)

cpc = pd.read_excel("dados/CPC_2023.xlsx", "CPC_2023"); cpc.columns = cpc.columns.str.strip()
cpc = cpc[cpc["Código da Área"].isin(AREAS_8) & cpc["Modalidade de Ensino"].eq("Educação Presencial")].copy()
cpc[ALVO] = 100 * cpc["Nota Bruta – Regime de Trabalho"]
cpc["Privada"] = ~cpc["Categoria Administrativa"].str.startswith("Pública")
ies = pd.read_csv("dados/MICRODADOS_ED_SUP_IES_2023.CSV", sep=";", encoding="latin1", low_memory=False)
ies = ies[ies["QT_DOC_EXE"] > 0].copy(); n = ies["QT_DOC_EXE"]
ies = ies.assign(**{"IES % integral": 100 * ies["QT_DOC_EX_INT"] / n, "IES % dedicação exclusiva": 100 * ies["QT_DOC_EX_INT_DE"] / n,
                    "IES % parcial": 100 * ies["QT_DOC_EX_PARC"] / n, "IES % doutores": 100 * ies["QT_DOC_EX_DOUT"] / n,
                    "IES % mestres": 100 * ies["QT_DOC_EX_MEST"] / n, "IES % 60+ anos": 100 * ies["QT_DOC_EX_60_MAIS"] / n,
                    "IES docentes (log)": np.log10(n), "IES técnicos por docente": ies["QT_TEC_TOTAL"] / n,
                    "Universidade": ies["TP_ORGANIZACAO_ACADEMICA"].eq(1).astype(int), "Comunitária": ies["IN_COMUNITARIA"].fillna(0).astype(int),
                    "Sem fins lucrativos": ies["TP_CATEGORIA_ADMINISTRATIVA"].eq(5).astype(int), "Capital": ies["IN_CAPITAL_IES"].fillna(0).astype(int)})
ATR_IES = ["IES % integral", "IES % dedicação exclusiva", "IES % parcial", "IES % doutores", "IES % mestres", "IES % 60+ anos",
           "IES docentes (log)", "IES técnicos por docente", "Universidade", "Comunitária", "Sem fins lucrativos", "Capital"]
cursos = cpc.merge(ies.set_index("CO_IES")[ATR_IES], left_on="Código da IES", right_index=True, how="inner")
COLS_AREA = [f"Área {a}" for a in AREAS_8]
for a, c in zip(AREAS_8, COLS_AREA):
    cursos[c] = cursos["Código da Área"].eq(a).astype(int)
ATR = ATR_IES + COLS_AREA
sup = cursos[cursos["Privada"] & cursos["Código da IES"].ne(PUC)].dropna(subset=ATR + [ALVO]).reset_index(drop=True)
puc = cursos[cursos["Código da IES"].eq(PUC) & cursos["Código da Área"].isin(FOCO)].copy()

GRADE = [(f, m) for f in [1, 5, 10, 20] for m in [1.0, 0.33]]
fab_rf = lambda p: RandomForestRegressor(n_estimators=300, min_samples_leaf=p[0], max_features=p[1], random_state=42, n_jobs=-1)
fab_lin = lambda: Pipeline([("e", StandardScaler()), ("m", LinearRegression())])

qual, prev_puc, resid_teste = [], {m: [] for m in ("Regressão linear", "Random Forest")}, []
for s in SEMENTES:
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=s).split(sup, groups=sup["Código da IES"]))
    Xtr, ytr, Xte, yte, gtr = sup.loc[tr, ATR], sup.loc[tr, ALVO], sup.loc[te, ATR], sup.loc[te, ALVO], sup.loc[tr, "Código da IES"]
    melhor = min(GRADE, key=lambda p: mean_absolute_error(ytr, cross_val_predict(fab_rf(p), Xtr, ytr, groups=gtr, cv=GroupKFold(5))))
    modelos = {"Média do treino": None, "Regressão linear": fab_lin().fit(Xtr, ytr), "Random Forest": fab_rf(melhor).fit(Xtr, ytr)}
    for nome, m in modelos.items():
        p_te = np.full(len(te), ytr.mean()) if m is None else m.predict(Xte)
        p_tr = np.full(len(tr), ytr.mean()) if m is None else m.predict(Xtr)
        qual.append({"Modelo": nome, "Semente": s, "R² treino": r2_score(ytr, p_tr), "R² teste": r2_score(yte, p_te),
                     "MAE teste": mean_absolute_error(yte, p_te), "Parâmetros": str(melhor) if nome == "Random Forest" else ""})
        if m is not None:
            prev_puc[nome].append(m.predict(puc[ATR]))
        if nome == "Random Forest":
            resid_teste.extend(yte - p_te)
q = pd.DataFrame(qual)
res_q = q.groupby("Modelo", sort=False).agg(**{"R² treino": ("R² treino", "mean"), "R² teste": ("R² teste", "mean"),
                                               "R² teste (desvio)": ("R² teste", "std"), "MAE teste": ("MAE teste", "mean")}).reset_index()
res_q.round(3).to_csv(SAIDA / "rf_qualidade.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")

resid_teste = np.array(resid_teste)
linhas = []
for i, (_, c) in enumerate(puc.iterrows()):
    rf = np.array([p[i] for p in prev_puc["Random Forest"]]); lin = np.array([p[i] for p in prev_puc["Regressão linear"]])
    area = sup[sup["Código da Área"] == c["Código da Área"]][ALVO]
    linhas.append({"Curso": FOCO[c["Código da Área"]], "Real (%)": c[ALVO], "Esperado RF (%)": rf.mean(), "RF mín.": rf.min(), "RF máx.": rf.max(),
                   "Esperado linear (%)": lin.mean(), "Resíduo RF (p.p.)": c[ALVO] - rf.mean(),
                   "Percentil do resíduo": 100 * (resid_teste < c[ALVO] - rf.mean()).mean(), "Média dos privados da área (%)": area.mean()})
res_p = pd.DataFrame(linhas)
res_p.round(2).to_csv(SAIDA / "rf_puc.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print("cursos privados na base:", len(sup), "| IES:", sup["Código da IES"].nunique())
print(res_q.round(3).to_string(index=False)); print(res_p.round(1).to_string(index=False))

# figura do artigo: real x esperado da PUC-Rio, com a faixa das 10 divisões e a média dos privados da área
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
CM = 1 / 2.54
fig, ax = plt.subplots(figsize=(16 * CM, 5.6 * CM), dpi=300)
for i, r in res_p.iterrows():
    y = len(res_p) - 1 - i
    ax.plot([r["RF mín."], r["RF máx."]], [y, y], color="#7A8BA6", lw=4, solid_capstyle="round", alpha=0.6, label="Faixa do esperado nas 10 divisões" if i == 0 else None)
    ax.scatter(r["Esperado RF (%)"], y, s=34, color="#112B5C", zorder=3, label="Esperado pelo perfil da instituição (RF)" if i == 0 else None)
    ax.scatter(r["Real (%)"], y, marker="D", s=30, color="#EB6834", zorder=4, label="Real da PUC-Rio" if i == 0 else None)
    ax.scatter(r["Média dos privados da área (%)"], y, marker="|", s=160, color="#52514E", zorder=3, label="Média dos cursos privados da área" if i == 0 else None)
    ax.annotate(f"{r['Resíduo RF (p.p.)']:+.0f} p.p.".replace("-", "−"), ((r["Real (%)"] + r["Esperado RF (%)"]) / 2, y + 0.22),
                ha="center", fontsize=7, color="#EB6834")
ax.set_yticks(range(len(res_p))); ax.set_yticklabels(res_p["Curso"][::-1])
ax.set_xlim(20, 100); ax.set_xlabel("Docentes não horistas do curso (%)"); ax.set_ylim(-0.6, len(res_p) - 0.3)
ax.grid(axis="x", color="#E4E3DF", lw=0.6); ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.38), ncol=2, frameon=False, fontsize=6.8)
fig.tight_layout()
fig.savefig(r"diagramas\diagramas_fase3\artigo_fig7_regime_esperado.png", dpi=300, facecolor="white", bbox_inches="tight")
print("figura ok")

# registro no MLflow: uma execução-mãe com o resumo e uma filha por divisão e modelo (treino e teste)
import sys, time
sys.path.insert(0, str(Path(__file__).parent))
from registro_mlflow import iniciar
mlflow = iniciar()
with mlflow.start_run(run_name="Random Forest · Regime esperado", tags={"modelo": "RandomForestRegressor", "etapa": "diagnostico",
                                                                        "data": time.strftime("%Y-%m-%d %H:%M")}):
    mlflow.log_params({"alvo": ALVO, "base": "cursos presenciais de IES privadas das 8 areas, sem a PUC-Rio", "cursos": len(sup),
                       "ies": int(sup["Código da IES"].nunique()), "atributos": ", ".join(ATR), "divisoes": len(SEMENTES),
                       "sementes": "42-51", "teste": "20% das IES (GroupShuffleSplit)", "selecao_parametros": "GroupKFold(5) no treino, MAE",
                       "grade": str(GRADE), "n_estimators": 300})
    for _, r in res_q.iterrows():
        tag = {"Média do treino": "media", "Regressão linear": "linear", "Random Forest": "rf"}[r["Modelo"]]
        mlflow.log_metrics({f"{tag}_r2_treino": r["R² treino"], f"{tag}_r2_teste": r["R² teste"],
                            f"{tag}_r2_teste_desvio": r["R² teste (desvio)"], f"{tag}_mae_teste": r["MAE teste"]})
    for _, r in res_p.iterrows():
        tag = {"Engenharia Civil": "civil", "Engenharia de Produção": "producao", "Arquitetura e Urbanismo": "arquitetura"}[r["Curso"]]
        mlflow.log_metrics({f"puc_{tag}_real": r["Real (%)"], f"puc_{tag}_esperado": r["Esperado RF (%)"],
                            f"puc_{tag}_residuo": r["Resíduo RF (p.p.)"], f"puc_{tag}_percentil_residuo": r["Percentil do resíduo"]})
    for _, r in q.iterrows():
        with mlflow.start_run(run_name=f"{r['Modelo']} · semente {r['Semente']}", nested=True, tags={"nivel": "divisao"}):
            mlflow.log_params({"modelo": r["Modelo"], "semente": int(r["Semente"]), "parametros_rf": r["Parâmetros"] or "-"})
            mlflow.log_metrics({"r2_treino": r["R² treino"], "r2_teste": r["R² teste"], "mae_teste": r["MAE teste"]})
    for arq in (SAIDA / "rf_qualidade.csv", SAIDA / "rf_puc.csv", Path(r"diagramas\diagramas_fase3\artigo_fig7_regime_esperado.png")):
        mlflow.log_artifact(str(arq))
    q.to_csv(SAIDA / "rf_qualidade_por_divisao.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
    mlflow.log_artifact(str(SAIDA / "rf_qualidade_por_divisao.csv"))
    import mlflow.sklearn
    final = fab_rf(melhor).fit(sup[ATR], sup[ALVO])   # modelo para replicação: treinado em toda a base com os parâmetros da última divisão
    mlflow.sklearn.log_model(final, name="rf_regime_esperado", input_example=sup[ATR].head(3).astype(float),
                             skops_trusted_types=["sklearn.tree._tree.Tree"])
print("MLflow ok")

