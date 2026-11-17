"""Docentes por regime de trabalho (Painel Estatístico do Censo da Educação Superior, Inep — página Docente),
filtrados por instituição e área do curso (classificação Cine), copiados em 13/11/2026 com "Mostrar como uma tabela".
O painel soma todos os cursos da instituição naquela área (todos os campi). Uso só descritivo; não entra em modelo.
Saída: resultados_fase3/painel_docentes_regime.csv (série 2010-2024) e painel_docentes_regime_2023.csv."""
import pandas as pd

BRUTO = {
    ("Engenharia Civil", "PUC-Rio"): "Ano|Tempo Integral com DE|Tempo Integral sem DE|Tempo Parcial|Horista|2010|43||12|24|2011||49|10|22|2012||31|6|32|2013||30|2|35|2014||45|7|34|2015||59|5|31|2016||67|2|13|2017||68|7|27|2018||62|7|37|2019||53|7|16|2020||54|4|34|2021||29|1|11|2022||40|3|34|2023||51|6|35|2024||38|5|30",
    ("Engenharia Civil", "UNISINOS"): "Ano|Tempo Integral com DE|Tempo Integral sem DE|Tempo Parcial|Horista|2010|4|17|44|7|2011||22|41|7|2012||21|52|4|2013||21|53|6|2014||29|45|7|2015||29|50|7|2016||32|65|9|2017||34|85|5|2018|3|38|83|5|2019||45|79|7|2020|1|49|78|5|2021||59|68|7|2022||55|59|7|2023||44|60|6|2024||38|49|12",
    ("Engenharia Civil", "UCDB"): "Ano|Tempo Integral sem DE|Tempo Parcial|Horista|2011|6|4||2012|7|8|2|2013|11|14|4|2014|22|15|2|2015|20|17|1|2016|15|17|2|2017|23|13|2|2018|19|16|3|2019|25|19||2020|15|18|1|2021|8|24|1|2022|17|14|5|2023|15|23|2|2024|5|33|2",
    ("Engenharia Civil", "UP"): "Ano|Tempo Integral sem DE|Tempo Parcial|Horista|2010|22|5|4|2011|17|8|4|2012|21|20|8|2013|13|21|12|2014|19|17|5|2015|27|14|22|2016|18|17|11|2017|33|20|8|2018|30|24|2|2019|31|13|5|2020|34|23|10|2021|26|23|4|2022|5|5|7|2023|28|5|4|2024|6|9|1",
    ("Engenharia Civil", "UVA"): "Ano|Tempo Integral sem DE|Tempo Parcial|Horista|2010|13|7|13|2011|13|5|7|2012|14|5|16|2013|11|6|20|2014|19|20|32|2015|21|17|44|2016|22|17|42|2017|38|18|35|2018|29|16|36|2019|28|17|32|2020|25|8|31|2021|25|9|26|2022|24|5|23|2023|17|6||2024|7|17|3",
    ("Engenharia de Produção", "PUC-Rio"): "Ano|Tempo Integral com DE|Tempo Integral sem DE|Tempo Parcial|Horista|2010|55|1|10|44|2011||52|9|35|2012||52|7|40|2013||74|7|36|2014||73|4|38|2015||72|9|39|2016||74|5|30|2017||71|9|27|2018||69|8|42|2019||60|7|21|2020||62|6|39|2021||20|1|42|2022||47|7|38|2023||50|7|53|2024||50|9|58",
    ("Engenharia de Produção", "MACKENZIE"): "Ano|Tempo Integral sem DE|Tempo Parcial|Horista|2010|16|6|49|2011|19|6|46|2012|28|7|44|2013|30|6|49|2014|50|19|45|2015|68|30|30|2016|74|35|28|2017|108|42|56|2018|127|51|58|2019|130|72|49|2020|41|35|44|2021|26|37|29|2022|34|34|15|2023|48|32|15|2024|42|40|19",
    ("Engenharia de Produção", "PUCPR"): "Ano|Tempo Integral sem DE|Tempo Parcial|Horista|2010|25|18|32|2011|27|17|49|2012|34|10|64|2013|28|18|71|2014|38|71|36|2015|33|49|68|2016|40|51|55|2017|42|54|36|2018|42|42|32|2019|48|78|9|2020|37|54|25|2021|38|41|10|2022|48|46|10|2023|62|91|3|2024|52|65|8",
    ("Engenharia de Produção", "PUC GOIÁS"): "Ano|Tempo Integral sem DE|Tempo Parcial|Horista|2010|41|1|26|2011|45|2|72|2012|35|4|27|2013|23|2|41|2014|36|3|29|2015|36|3|70|2016|22|2|30|2017|36|11|10|2018|31|9|14|2019|28|12|11|2020|24|5|10|2021|26|11|14|2022|23|13|14|2023|18|16|11|2024|22|7|14",
    ("Arquitetura e Urbanismo", "PUC-Rio"): "Ano|Tempo Integral com DE|Tempo Integral sem DE|Tempo Parcial|Horista|2010|15||15|52|2011||8|13|55|2012||11|9|72|2013||10|12|74|2014||14|10|67|2015||11|9|64|2016||14|11|63|2017||21|5|55|2018||20|4|64|2019||19|6|60|2020||20|6|63|2021||14|2|49|2022||15|4|52|2023||20|5|60|2024||23|9|59",
    ("Arquitetura e Urbanismo", "UGB"): "Ano|Tempo Integral sem DE|Tempo Parcial|Horista|2010|6|12|1|2011|5|12|1|2012|5|15|1|2013|5|13|2|2014|6|21|1|2015|7|25||2016|6|25||2017|7|30||2018|6|24||2019|6|24||2020|5|26|1|2021|7|26|1|2022|6|25|1|2023|7|15||2024|9|13|2",
    ("Arquitetura e Urbanismo", "UNIP"): "Ano|Tempo Integral sem DE|Tempo Parcial|Horista|2010|164|75|14|2011|241|82|13|2012|308|86|14|2013|364|123|33|2014|334|143|85|2015|428|148|108|2016|436|110|95|2017|396|124|70|2018|382|100|80|2019|322|108|62|2020|315|104|46|2021|200|114|61|2022|207|79|58|2023|134|61|40|2024|107|41|50",
    ("Arquitetura e Urbanismo", "UNIMAR"): "Ano|Tempo Integral sem DE|Tempo Parcial|Horista|2010|6|8|3|2011|7|13||2012|3|12|1|2013|3|13|1|2014|5|14||2015|3|13|1|2016|3|15||2017|8|12||2018|1|12|2|2019|5|11|2|2020|1|10||2021|1|10||2022|1|8||2023|1|8||2024|1|8|",
}
linhas = []
for (area, ies), s in BRUTO.items():
    p = s.split("|"); i = p.index("Ano"); cab = []
    while not p[i].isdigit():
        cab.append(p[i]); i += 1
    vals = p[i:]
    for k in range(0, len(vals) - len(cab) + 1, len(cab)):
        reg = dict(zip(cab, vals[k:k + len(cab)]))
        linhas.append({"Área": area, "IES": ies, "Ano": int(reg["Ano"]),
                       **{c: int(reg.get(c) or 0) for c in ["Tempo Integral com DE", "Tempo Integral sem DE", "Tempo Parcial", "Horista"]}})
d = pd.DataFrame(linhas)
d["Tempo integral"] = d["Tempo Integral com DE"] + d["Tempo Integral sem DE"]
d["Total"] = d["Tempo integral"] + d["Tempo Parcial"] + d["Horista"]
d["% horistas"] = 100 * d["Horista"] / d["Total"]
d.to_csv("resultados_fase3/painel_docentes_regime.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
d23 = d[d.Ano == 2023][["Área", "IES", "Tempo integral", "Tempo Parcial", "Horista", "Total", "% horistas"]]
d23.round(1).to_csv("resultados_fase3/painel_docentes_regime_2023.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
print(d23.round(1).to_string(index=False))
