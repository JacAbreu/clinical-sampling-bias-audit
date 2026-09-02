# Exploração de dados clínicos — base FAPESP COVID-19 Data Sharing/BR

## Contexto

Este diretório reúne os scripts e relatórios de exploração de dado gerados em
2026-07-06, durante a fase de ajuste do projeto **MOSAIC-FL** (aprendizado
federado + RAG para predição de trajetória clínica — TCC do MBA em
Inteligência Artificial e Big Data, ICMC/USP). O código de produção do
MOSAIC-FL vive em outro repositório — **[JacAbreu/mosaic-fl](https://github.com/JacAbreu/mosaic-fl)**
—; este aqui é o laboratório onde perguntas sobre a base de dado real
(**[FAPESP COVID-19 Data Sharing/BR](https://drive.usercontent.google.com/download?id=1CwUMQuS-aBs3as_KMxED2gRWS1MOw-Fp&export=download&authuser=0)**)
foram investigadas antes de qualquer decisão de arquitetura ser tomada.

Duas perguntas concretas motivaram este trabalho:

1. **O esquema de 5 classes de prognóstico usado no MOSAIC-FL é o melhor
   possível?** As classes atuais (`curado_pronto`, `curado_internado`,
   `melhora_pronto`, `melhora_internado_breve`, `melhora_internado_grave`)
   vêm de regra clínica manual, não de dado — inclusive um limiar de 10 dias
   (breve × grave) nunca validado estatisticamente. Existe um esquema de
   classe, descoberto por clusterização sobre o próprio dado, que separe
   melhor os desfechos e seja mais equilibrado entre hospitais?
2. **Que esquema de classe sobrevive melhor à privacidade diferencial?** Uma
   vez que ficou claro que o DP-FedAvg tem custo de acurácia severo (achado
   documentado no trabalho principal), a pergunta virou: existe um esquema de
   classe cuja acurácia caia pouco sob ruído de DP, mesmo que não seja o
   esquema com a melhor acurácia bruta?

Vários achados aqui foram incorporados, depois, à fundamentação do trabalho
principal — o limiar de 10 dias não validado e o desbalanceamento
(*label skew*) extremo entre hospitais, em particular, vieram diretamente
desta investigação.

## Duas linhas de trabalho

### 1. Auditoria de qualidade e viés de amostragem

Arquivos: `exploracao-dados-fapesp.py`, `exploracao-dados-fapesp-dependencias-novas.py`,
`auditoria_*.txt`, `analises_*.txt`, `relatorio*.html` (perfilamento via
`ydata_profiling`).

Achado mais relevante: **viés de granularidade na amostragem**. Amostrar
linhas de *exame* (em vez de atendimentos) favorece atendimentos com muitos
exames, distorcendo a distribuição real de desfecho — um atendimento com 300
exames tem 300× mais chance de entrar na amostra do que um com 1 exame. A
correção, visível na evolução v3 → v4 dos scripts de *pipeline* abaixo, passou
a amostrar *atendimentos* distintos (chance igual para todos) e trazer todos
os exames de cada um, eliminando esse viés.

### 2. Descoberta de esquema de classe (clusterização + robustez a DP)

Arquivos: `exploracao-database-fapesp-pipeline-mosaic-v2.py` a `v5.py`,
`pipeline-mosaic-v4.py` a `v6.py`, `clusters_*.csv`, `vies_*.txt`.

Cada execução gera arquivos versionados por data/hora (relatório, *clusters*,
auditoria, análises), para acumular extrações e comparar visões ao longo do
tempo. Evolução das versões:

- **v2** — primeira tentativa de descobrir esquema de classe por
  clusterização (`k-prototypes`, que lida com variável numérica e categórica
  ao mesmo tempo) e comparar contra as 5 classes teóricas atuais.
- **v3 → v4** — correção do viés de granularidade descrito acima (amostragem
  por atendimento, não por linha de exame).
- **v5 → v6** — adiciona uma métrica direta de robustez à privacidade
  diferencial: simula ruído gaussiano calibrado por *epsilon* (amplificado
  para classe rara, reproduzindo o colapso observado sob DP) e mede a
  acurácia balanceada resultante por esquema de classe, sem precisar treinar
  o modelo pesado — permite comparar esquemas pelo critério que de fato
  importa (acurácia *com* privacidade, não só acurácia bruta).

## Status

Exploratório — resultado de investigação, não *pipeline* de produção. Nenhuma
decisão daqui foi incorporada automaticamente ao MOSAIC-FL: qualquer mudança
de esquema de classe motivada por este trabalho precisa ser decidida e
implementada explicitamente no repositório principal.

## Reproduzindo

Ambiente virtual em `explorar-dados-env/` (não deveria ser versionado —
recriar com `python -m venv explorar-dados-env`). Dependências: `pandas`,
`numpy`, `sqlalchemy`, `psycopg2-binary`, `ydata-profiling`, `kmodes`.

Os scripts esperam um banco PostgreSQL local com o schema do MOSAIC-FL
(`clinical.attendances`, `clinical.patients`, `metrics.clinical_outcomes`,
`metrics.exam_records`) já carregado.

**Atenção antes de publicar ou compartilhar este repositório**: alguns
scripts têm a *string* de conexão com uma credencial de desenvolvimento local
escrita diretamente no código (`postgresql://mosaicfl:...@localhost:5432/mosaicfl`).
É uma senha de banco local, não uma credencial de produção — mas, ainda
assim, mova isso para variável de ambiente antes de tornar o repositório
público.

## Estrutura

- `*.py` — scripts de exploração/*pipeline*, cada versão nomeada (`v2` a `v6`)
- `relatorio*.html` — relatórios de perfilamento (`ydata_profiling`), um por execução
- `clusters*.csv` — resultado de clusterização, um por execução
- `auditoria_*.txt` / `analises_*.txt` / `vies_*.txt` — saída textual de cada etapa de auditoria/análise
- `explorar-dados-env/` — ambiente virtual Python (não deveria ir para o controle de versão)
