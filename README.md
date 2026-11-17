# Com quem comparar? Diagnóstico dos cursos da PUC-Rio a partir dos indicadores do CPC 2023

Trabalho final da disciplina IND2622 — Ciência de Dados para Processos de Negócios (PUC-Rio, 2026).
Autora: Thaís da Costa Ramos.

## Pergunta

Com quem cada curso da PUC-Rio deve ser comparado dentro da sua área de avaliação, e em quais indicadores do
Conceito Preliminar de Curso (CPC) ele se diferencia desses pares?

## O que o projeto faz

1. **Agrupa os cursos por área** com K-means sobre os 8 indicadores do CPC 2023, em 8 cenários de dados (C1 a C8),
   com treino e teste repetidos em 10 divisões (sementes 42 a 51). O cenário é escolhido pelo ganho de silhueta sobre
   dados simulados sem grupos e pela estabilidade (ARI e Jaccard), e não só pela silhueta.
2. **Escolhe instituições de referência** com k-NN (`sklearn.neighbors.NearestNeighbors`): mesma área, mesmo grupo,
   privadas, com Regime de trabalho acima da média, e distância medida só no que a PUC-Rio não controla.
3. **Estima o Regime de trabalho esperado** com Random Forest a partir do perfil docente da instituição (Censo 2023),
   validado em 10 divisões com instituições inteiras no teste e comparado com a média e a regressão linear.
4. **Simula a meta**: quantos docentes horistas converter para chegar à média dos cursos privados da área.

Áreas aprofundadas: Engenharia Civil, Engenharia de Produção e Arquitetura e Urbanismo.

## Dados (públicos, Inep)

| Arquivo | Fonte |
|---|---|
| `dados/resultado_cpc_2017.csv`, `dados/resultados_cpc_2019.xlsx`, `dados/CPC_2023.xlsx` | Resultados do CPC — https://www.gov.br/inep/pt-br/areas-de-atuacao/pesquisas-estatisticas-e-indicadores/indicadores-de-qualidade-da-educacao-superior |
| `dados/microdados_censo_da_educacao_superior_2023.zip`, `dados/MICRODADOS_ED_SUP_IES_2023.CSV` | Microdados do Censo da Educação Superior 2023 — https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2023.zip |
| `scripts_fase3/painel_docentes_regime.py` (valores transcritos) | Painel estatístico do Censo da Educação Superior (Inep), página Docente |
| `dados/processados/base_analitica_1907_cursos.csv` | Base analítica gerada pelo pré-processamento |

## Ordem de execução

Notebooks (na raiz):

| Ordem | Arquivo | Conteúdo |
|---|---|---|
| 1 | `01_analise_exploratoria.ipynb` | Análise exploratória |
| 2 | `historico_preprocessamento_e_testes_ate_25_10.ipynb` | Pré-processamento e montagem da base analítica |
| 3 | `03_diagnostico_experimentos_CPC2023.ipynb` | Plano de experimentação |
| 4 | `04_influencia_de_k_na_silhueta_CPC2023.ipynb` | Influência de K na silhueta |
| 5 | `05_roteiro_comparativo_agrupamentos_CPC2023.ipynb`, `05_1_...`, `05_2_...` | Comparação dos cenários e validação |
| 6 | `06_diagnostico_PUCRio_por_area_CPC2023_REVISADO.ipynb`, `07_sintese_...` | Diagnóstico da PUC-Rio por área |
| 7 | `08_validacao_externa_e_algoritmos_CPC2023.ipynb` | Outros algoritmos e validação externa |
| 8 | `09_agrupamento_por_area_MLflow_CPC2023.ipynb` | Agrupamento das 8 áreas com registro no MLflow |
| 9 | `10_treino_teste_por_configuracao_CPC2023.ipynb` | Treino × teste por configuração |
| 10 | `11_referencias_e_regime_3cursos_CPC2023.ipynb` | Referências e regime nas 3 áreas |

Scripts da fase 3 (pasta `scripts_fase3/`, rodar a partir da raiz do repositório):

```
python scripts_fase3/knn_referencias.py        # k-NN: referências e estabilidade (MLflow)
python scripts_fase3/rf_regime_esperado.py     # Random Forest: regime esperado e resíduo (MLflow)
python scripts_fase3/painel_docentes_regime.py # composição docente do painel do Inep
python scripts_fase3/numeros_apresentacao.py   # diferenças para os pares, C2 × C5 e simulação da meta
python scripts_fase3/funil_knn.py              # funil do critério do k-NN
python scripts_fase3/regime_por_curso.py       # planilha do regime por curso
python scripts_fase3/graficos_apresentacao.py  # gráficos da apresentação
```

O plano experimental, com uma linha por execução, está em `resultados_etapa2/Plano_Experimento_Agrupamento_CPC2023.xlsx`.

## MLflow

Os experimentos estão em `mlflow_cpc.db` (agrupamento por área, k-NN e Random Forest, com uma execução por divisão)
e `mlartifacts/`. Para ver:

```
mlflow ui --backend-store-uri sqlite:///mlflow_cpc.db
```

Ao rodar os scripts de novo, novas execuções são registradas no mesmo banco.

## Ambiente

Python 3.14. Bibliotecas em `requirements.txt` (`pip install -r requirements.txt`).

## Uso de IA

Utilizou-se assistente de IA (Claude) como apoio na programação dos scripts.
