"""Gráficos da apresentação da fase 3 (matplotlib, letra grande para projeção). Saída: resultados_fase3/apresentacao/*.png"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

OUT = Path("resultados_fase3/apresentacao"); OUT.mkdir(parents=True, exist_ok=True)
NAVY, AZUL, CLARO, LARANJA, CINZA, TXT, GRADE = "#112B5C", "#2A78D6", "#A9C4EB", "#EB6834", "#B8C0CC", "#1F2A44", "#E4E3DF"
plt.rcParams.update({"font.family": "Arial", "font.size": 17, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9AA3B2", "xtick.color": TXT, "ytick.color": TXT, "axes.labelcolor": TXT})
AREAS = ["Engenharia Civil", "Engenharia de Produção", "Arquitetura e Urbanismo"]
CURTO = {"Engenharia Civil": "Civil", "Engenharia de Produção": "Produção", "Arquitetura e Urbanismo": "Arquitetura"}
virg = lambda v, c=2: f"{v:.{c}f}".replace(".", ",").replace("-", "−")
def salva(fig, nome):
    fig.savefig(OUT / nome, dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)

dif = pd.read_csv("resultados_fase3/diferencas_puc_pares_C2.csv", sep=";", decimal=",", encoding="utf-8-sig")
ORDEM = ["Enade", "IDD", "Mestres", "Doutores", "Regime de trabalho", "Org. didático-pedag.", "Infraestrutura", "Oportunidades"]
CONTROLAVEL = {"Mestres", "Doutores", "Regime de trabalho"}
sim = pd.read_csv("resultados_fase3/simulacao_meta_regime.csv", sep=";", decimal=",", encoding="utf-8-sig")

# A · o CPC esconde 8 indicadores (notas da PUC-Rio)
fig, axs = plt.subplots(1, 3, figsize=(20, 7.2), sharey=True)
for ax, a in zip(axs, AREAS):
    d = dif[dif["Área"] == a].set_index("Indicador").loc[ORDEM]
    cores = [NAVY if i in CONTROLAVEL else CLARO for i in ORDEM]
    ax.barh(range(8)[::-1], d["PUC-Rio"], color=cores, height=0.65)
    for y, v in zip(range(8)[::-1], d["PUC-Rio"]):
        ax.text(v + 0.08, y, virg(v), va="center", fontsize=15, color=TXT)
    s = sim[sim["Área"] == a].iloc[0]
    ax.set_title(f"{CURTO[a]}\nCPC {virg(s['CPC contínuo atual'])} · faixa {int(s['CPC faixa atual'])}", fontsize=21, fontweight="bold", color=NAVY)
    ax.set_xlim(0, 5.6); ax.set_xticks(range(6)); ax.grid(axis="x", color=GRADE); ax.set_axisbelow(True)
axs[0].set_yticks(range(8)[::-1]); axs[0].set_yticklabels(ORDEM, fontsize=17)
fig.legend(handles=[Patch(color=NAVY, label="Sob controle da gestão"), Patch(color=CLARO, label="Depende dos estudantes")],
           loc="upper center", ncol=2, frameon=False, fontsize=17, bbox_to_anchor=(0.5, 1.06))
fig.text(0.5, -0.03, "Nota padronizada do indicador (0 a 5), CPC 2023", ha="center", fontsize=16, color=TXT)
salva(fig, "a_cpc_decomposto.png")

# B · a silhueta engana (Engenharia Civil, 100% do treino)
fig, (e1, e2) = plt.subplots(1, 2, figsize=(15, 6.2))
cen = ["C1 · CPC global", "C4 · PCA 2D", "C2 · 8 indicadores"]
for ax, vals, tit, lim in ((e1, [0.591, 0.403, 0.258], "Silhueta no teste", 0.7), (e2, [-0.002, 0.031, 0.066], "Ganho sobre dados sem grupos", 0.08)):
    cores = [CINZA, CINZA, NAVY]
    ax.bar(cen, vals, color=cores, width=0.6)
    for x, v in enumerate(vals):
        ax.text(x, max(v, 0) + lim * 0.02, virg(v, 3), ha="center", fontsize=18, fontweight="bold", color=TXT)
    ax.set_title(tit, fontsize=21, fontweight="bold", color=NAVY); ax.set_ylim(min(0, min(vals)) - lim * 0.05, lim)
    ax.axhline(0, color="#9AA3B2", lw=1); ax.tick_params(axis="x", labelsize=16); ax.grid(axis="y", color=GRADE); ax.set_axisbelow(True)
salva(fig, "b_silhueta_engana.png")

# C · quais áreas têm grupos reais e estáveis
ar = pd.DataFrame({"Área": ["Civil", "Produção", "Arquitetura", "Mecânica", "Elétrica", "Química", "Ambiental", "Controle e Aut."],
                   "Cursos": [451, 300, 266, 245, 234, 107, 109, 93], "Ganho": [0.066, 0.053, 0.053, 0.023, 0.027, 0.037, -0.012, 0.006],
                   "ARI": [0.95, 0.83, 0.94, 0.87, 0.82, 0.87, 0.60, 0.75], "Jaccard": [0.97, 0.85, 0.94, 0.42, 0.51, 0.60, 0.34, 0.89]})
ar["Atende"] = (ar.Ganho >= 0.05) & (ar.ARI >= 0.80) & (ar.Jaccard >= 0.85)
fig, ax = plt.subplots(figsize=(14, 7.5))
ax.axvspan(0.05, 0.08, ymin=(0.80 - 0.55) / 0.47, ymax=1, color="#E8F0FB", zorder=0)
ax.axvline(0.05, color=LARANJA, ls="--", lw=1.6); ax.axhline(0.80, color=LARANJA, ls="--", lw=1.6)
for _, r in ar.iterrows():
    ax.scatter(r.Ganho, r.ARI, s=r.Cursos * 4, color=NAVY if r.Atende else CINZA, edgecolor="white", lw=1.5, zorder=3)
    dx = {"Civil": (0.0, -0.06), "Produção": (0.003, -0.03), "Arquitetura": (-0.0025, 0.03), "Elétrica": (-0.004, -0.035), "Química": (0.003, 0.01)}.get(r.Área, (0.003, 0.012))
    ax.text(r.Ganho + dx[0], r.ARI + dx[1], f"{r.Área}\nJaccard {virg(r.Jaccard)}", fontsize=14, color=NAVY if r.Atende else "#6B7280",
            fontweight="bold" if r.Atende else "normal", zorder=4)
ax.set_xlim(-0.02, 0.08); ax.set_ylim(0.55, 1.02)
ax.set_xlabel("Ganho sobre dados sem grupos (grupos reais →)", fontsize=17); ax.set_ylabel("ARI entre as 10 divisões (estáveis ↑)", fontsize=17)
ax.text(0.0795, 1.012, "grupos reais e estáveis", color=LARANJA, fontsize=15, va="top", ha="right")
ax.grid(color=GRADE); ax.set_axisbelow(True)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: virg(v, 2)))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: virg(v, 2)))
salva(fig, "c_areas_ganho_ari.png")

# D · diferença PUC-Rio − pares do grupo C2
fig, axs = plt.subplots(1, 3, figsize=(20, 7.2), sharey=True)
for ax, a in zip(axs, AREAS):
    d = dif[dif["Área"] == a].set_index("Indicador").loc[ORDEM]
    cores = [LARANJA if i == "Regime de trabalho" else (AZUL if v >= 0 else "#8FA3BF") for i, v in zip(ORDEM, d["Diferença"])]
    ax.barh(range(8)[::-1], d["Diferença"], color=cores, height=0.65)
    for y, v in zip(range(8)[::-1], d["Diferença"]):
        ax.text(v + (0.1 if v >= 0 else -0.1), y, virg(v, 1), va="center", ha="left" if v >= 0 else "right", fontsize=15, color=TXT)
    ax.axvline(0, color=TXT, lw=1.2); ax.set_xlim(-3.2, 2.3)
    ax.set_title(CURTO[a], fontsize=22, fontweight="bold", color=NAVY); ax.grid(axis="x", color=GRADE); ax.set_axisbelow(True)
axs[0].set_yticks(range(8)[::-1]); axs[0].set_yticklabels(ORDEM, fontsize=17)
fig.text(0.5, -0.03, "← PUC-Rio abaixo dos pares        diferença de nota (PUC-Rio − média dos pares do grupo)        PUC-Rio acima →", ha="center", fontsize=16, color=TXT)
salva(fig, "d_diferencas_pares.png")

# E · Regime da PUC-Rio, dos pares no C2 e dos pares sem o Regime (C5)
reg = pd.read_csv("resultados_fase3/regime_pares_C2_C5.csv", sep=";", decimal=",", encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(14, 6.8)); x = np.arange(3); w = 0.26
for k, (col, cor, rot) in enumerate([("PUC-Rio", LARANJA, "PUC-Rio"), ("Pares C2", CLARO, "Pares com os 8 indicadores (C2)"), ("Pares C5", NAVY, "Pares sem o Regime (C5)")]):
    v = reg[col].to_numpy()
    ax.bar(x + (k - 1) * w, v, w * 0.92, color=cor, label=rot)
    for xi, vi in zip(x + (k - 1) * w, v):
        ax.text(xi, vi + 0.06, virg(vi, 1), ha="center", fontsize=17, fontweight="bold", color=TXT)
ax.set_xticks(x); ax.set_xticklabels([CURTO[a] for a in reg["Área"]], fontsize=19)
ax.set_ylim(0, 5.6); ax.set_ylabel("Nota de Regime de trabalho (0 a 5)", fontsize=17)
ax.legend(frameon=False, fontsize=16, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13))
ax.grid(axis="y", color=GRADE); ax.set_axisbelow(True)
salva(fig, "e_regime_c2_c5.png")

# G · composição do corpo docente (painel Inep, 2023: instituição × área do curso)
pnl = pd.read_csv("resultados_fase3/painel_docentes_regime_2023.csv", sep=";", decimal=",", encoding="utf-8-sig")
NOMEIES = {"PUC-Rio": "PUC-Rio", "UNISINOS": "Unisinos", "UCDB": "UCDB", "UP": "UP", "UVA": "UVA", "MACKENZIE": "Mackenzie", "PUCPR": "PUCPR",
           "PUC GOIÁS": "PUC Goiás", "UGB": "UGB", "UNIP": "UNIP", "UNIMAR": "Unimar"}
fig, axs = plt.subplots(1, 3, figsize=(20, 6.6), sharex=True)
for ax, a in zip(axs, AREAS):
    d = pnl[pnl["Área"] == a].copy()
    d = pd.concat([d[d.IES == "PUC-Rio"], d[d.IES != "PUC-Rio"]])[::-1]
    tot = d["Total"]; y = np.arange(len(d)); esq = np.zeros(len(d))
    for col, cor, rot in (("Tempo integral", NAVY, "Tempo integral"), ("Tempo Parcial", AZUL, "Tempo parcial"), ("Horista", LARANJA, "Horista")):
        v = 100 * d[col].to_numpy() / tot.to_numpy()
        ax.barh(y, v, left=esq, color=cor, height=0.62, label=rot)
        for yi, l, vi in zip(y, esq, v):
            if vi >= 9:
                ax.text(l + vi / 2, yi, f"{vi:.0f}%", ha="center", va="center", color="white", fontsize=14, fontweight="bold")
        esq += v
    ax.set_yticks(y); ax.set_yticklabels([NOMEIES[i] for i in d["IES"]], fontsize=17)
    for t in ax.get_yticklabels():
        if t.get_text() == "PUC-Rio":
            t.set_fontweight("bold"); t.set_color(LARANJA)
    ax.set_title(CURTO[a], fontsize=22, fontweight="bold", color=NAVY); ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100])
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=17, bbox_to_anchor=(0.5, 1.07))
fig.text(0.5, -0.03, "% dos docentes de todos os cursos da instituição na área (painel do Inep, 2023)", ha="center", fontsize=16, color=TXT)
fig.tight_layout()
salva(fig, "g_composicao_docente.png")

# I · simulação da meta
fig, ax = plt.subplots(figsize=(14, 6.8)); x = np.arange(3); w = 0.34
atual, meta = sim["CPC contínuo atual"].to_numpy(), (sim["CPC contínuo atual"] + sim["Ganho no CPC contínuo"]).to_numpy()
ax.bar(x - w / 2, atual, w * 0.92, color=CLARO, label="CPC contínuo atual")
ax.bar(x + w / 2, meta, w * 0.92, color=NAVY, label="Com o Regime na média dos privados da área")
for xi, a_, m_, n_ in zip(x, atual, meta, sim["Horistas a converter"]):
    ax.text(xi - w / 2, a_ + 0.04, virg(a_), ha="center", fontsize=17, color=TXT)
    ax.text(xi + w / 2, m_ + 0.04, virg(m_), ha="center", fontsize=17, fontweight="bold", color=NAVY)
    ax.text(xi, 1.25, f"converter\n{n_} horistas", ha="center", fontsize=16, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc=LARANJA, ec="none"))
for lim, rot in ((2.945, "faixa 4"), (3.945, "faixa 5")):
    ax.axhline(lim, color=LARANJA, ls="--", lw=1.4); ax.text(2.62, lim + 0.03, rot, color=LARANJA, fontsize=15, ha="right")
ax.set_xticks(x); ax.set_xticklabels([CURTO[a] for a in sim["Área"]], fontsize=19); ax.set_ylim(1, 4.3)
ax.set_ylabel("CPC contínuo", fontsize=17); ax.legend(frameon=False, fontsize=16, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
ax.grid(axis="y", color=GRADE); ax.set_axisbelow(True)
salva(fig, "i_simulacao_meta.png")
print("ok", sorted(p.name for p in OUT.glob("*.png")))


