"""Regime de trabalho da PUC-Rio e das referências estáveis do k-NN, por curso.
% de horistas: CPC 2023 (nota bruta do Regime = % de docentes não horistas do curso) — é o valor exato do curso.
Divisão entre tempo integral e parcial: só existe no painel do Inep, que soma todos os cursos da instituição na área
(todos os campi); por isso a parte não horista do curso é repartida na proporção integral/parcial do painel.
Saída: resultados_fase3/regime_por_curso.csv e .xlsx."""
import numpy as np, pandas as pd

ref = pd.read_csv("resultados_fase3/knn_referencias.csv", sep=";", decimal=",", encoding="utf-8-sig")
pnl = pd.read_csv("resultados_fase3/painel_docentes_regime_2023.csv", sep=";", decimal=",", encoding="utf-8-sig")
ref = ref[(ref["Papel"] == "PUC-Rio") | (ref["% das 10 divisões"] >= 50)].copy()
ref["IES"] = ref["Sigla da IES"].str.upper().replace({"PUC-RIO": "PUC-Rio"})
d = ref.merge(pnl, on=["Área", "IES"], how="left")
d["% horistas (CPC, curso)"] = 100 - d["% não horista (curso)"]
d["% horistas (painel, IES na área)"] = d["% horistas"]
d["Diferença painel − CPC (p.p.)"] = d["% horistas (painel, IES na área)"] - d["% horistas (CPC, curso)"]
fr = d["Tempo integral"] / (d["Tempo integral"] + d["Tempo Parcial"])
d["% integral (estimado no curso)"] = d["% não horista (curso)"] * fr
d["% parcial (estimado no curso)"] = d["% não horista (curso)"] * (1 - fr)
d["Painel"] = np.where(d["Total"].isna(), "sem dado coletado",
                       np.where(d["Diferença painel − CPC (p.p.)"].abs() <= 2, "bate com o curso", "soma outros cursos/campi"))
cols = ["Área", "Papel", "Sigla da IES", "Nome da IES", "Sigla da UF", "% das 10 divisões", "Distância",
        "Nota Padronizada - Regime de Trabalho", "% horistas (CPC, curso)", "% integral (estimado no curso)", "% parcial (estimado no curso)",
        "Tempo integral", "Tempo Parcial", "Horista", "Total", "% horistas (painel, IES na área)", "Diferença painel − CPC (p.p.)", "Painel"]
out = d[cols].rename(columns={"Tempo integral": "Painel: integral", "Tempo Parcial": "Painel: parcial", "Horista": "Painel: horistas",
                              "Total": "Painel: total", "Nota Padronizada - Regime de Trabalho": "Nota de Regime (CPC)"})
out.round(1).to_csv("resultados_fase3/regime_por_curso.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
# todos os cursos da mesma área de cada instituição no CPC 2023 (para comparar com o painel, que soma esses cursos)
cpc = pd.read_excel("dados/CPC_2023.xlsx", "CPC_2023"); cpc.columns = cpc.columns.str.strip()
AREA = {"Engenharia Civil": 5710, "Engenharia de Produção": 6208, "Arquitetura e Urbanismo": 21}
todos = []
for _, x in out.iterrows():
    s = cpc[(cpc["Sigla da IES"].str.upper() == x["Sigla da IES"].upper()) & (cpc["Código da Área"] == AREA[x["Área"]])]
    for _, y in s.iterrows():
        todos.append({"Área": x["Área"], "Sigla da IES": x["Sigla da IES"], "Código do Curso": y["Código do Curso"],
                      "Município": y["Município do Curso"], "Modalidade": y["Modalidade de Ensino"],
                      "% horistas (CPC)": round(100 - 100 * y["Nota Bruta – Regime de Trabalho"], 1) if pd.notna(y["Nota Bruta – Regime de Trabalho"]) else None})
todos = pd.DataFrame(todos)
with pd.ExcelWriter("resultados_fase3/regime_por_curso.xlsx", engine="openpyxl") as w:
    out.round(1).to_excel(w, sheet_name="Regime por curso", index=False)
    todos.to_excel(w, sheet_name="Todos os cursos no CPC", index=False)
    pd.DataFrame({"Nota": [
        "Fonte do % de horistas: CPC 2023 (Inep), nota bruta do Regime de Trabalho = proporção de docentes não horistas do curso.",
        "Fonte da divisão integral/parcial: painel estatístico do Censo da Educação Superior 2023 (Inep), filtro Instituição + Área Geral + Área do Curso.",
        "O painel soma todos os cursos da instituição naquela área (todos os campi); o integral/parcial do curso é estimado na proporção do painel.",
        "Referências: k-NN da fase 3, mantidas as que voltam em pelo menos 50% das 10 divisões."]}).to_excel(w, sheet_name="Leia-me", index=False)
    for ws in w.book.worksheets:
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(60, max(10, max(len(str(c.value or "")) for c in col) + 2))
pd.set_option("display.width", 250)
print(out[["Área", "Sigla da IES", "% das 10 divisões", "% horistas (CPC, curso)", "% integral (estimado no curso)", "% parcial (estimado no curso)",
           "% horistas (painel, IES na área)", "Painel"]].round(1).to_string(index=False))
