"""
=============================================================================
PIPELINE DE EXPLORAÇÃO E DESCOBERTA DE CLASSES — MOSAIC
=============================================================================
Roda em sequência:
  ETAPA 1 — Relatório visual de exploração (relatorio.html)
  ETAPA 2 — Clustering para descobrir classes de pacientes (k-prototypes)

Os dados são carregados UMA vez e reutilizados pelas duas etapas.

Dependências:
    pip install pandas sqlalchemy psycopg2-binary fg-data-profiling kmodes

Para rodar só uma etapa, ajuste os interruptores na CONFIGURAÇÃO abaixo.
=============================================================================

-----------------------------------------------------------------------------
CONCEITOS: por que k-prototypes, e não k-means puro?
-----------------------------------------------------------------------------

O QUE É CLUSTERING
    Clustering é aprendizado NÃO-SUPERVISIONADO: o algoritmo agrupa as linhas
    em "clusters" (grupos) de forma que itens parecidos fiquem juntos e itens
    diferentes fiquem separados — SEM que a gente diga de antemão quais são os
    grupos. É assim que "descobrimos classes" que não estavam rotuladas.
    Para decidir o que é "parecido", o algoritmo mede a DISTÂNCIA entre linhas.

K-MEANS (PURO)
    O k-means é o algoritmo de clustering mais conhecido. Ele funciona assim:
    coloca k "centros" no espaço, atribui cada ponto ao centro mais próximo, e
    recalcula os centros como a MÉDIA dos pontos de cada grupo — repetindo até
    estabilizar. A distância usada é a EUCLIDIANA (a "linha reta" entre dois
    pontos, como no teorema de Pitágoras).
    Isso funciona muito bem para dados NUMÉRICOS: faz sentido dizer que idade
    30 está mais perto de 35 do que de 80, e faz sentido tirar a média de idades.

O PROBLEMA COM DADOS CATEGÓRICOS
    Dados categóricos são valores sem ordem numérica: sexo (M/F), especialidade
    (cardiologia, pediatria...), município, tipo de atendimento, desfecho.
    O k-means NÃO funciona bem com eles por dois motivos:
      1. DISTÂNCIA não faz sentido. Qual é a "distância euclidiana" entre
         'cardiologia' e 'pediatria'? Não existe — categorias não são números.
         Um truque comum é converter categorias em números (0,1,2...), mas isso
         INVENTA uma ordem falsa: o algoritmo passaria a achar que
         cardiologia(0) está "mais perto" de pediatria(1) do que de
         ortopedia(2), o que é sem sentido clínico.
      2. MÉDIA não faz sentido. A média entre 'M' e 'F' não existe; a média
         entre 'cardiologia' e 'pediatria' também não. E o k-means depende de
         calcular médias para achar os centros dos grupos.

K-MODES (para dados categóricos)
    O k-modes resolve isso trocando as duas peças do k-means:
      - em vez de DISTÂNCIA euclidiana, conta QUANTOS ATRIBUTOS SÃO DIFERENTES
        entre duas linhas (ex.: se diferem em sexo e especialidade mas têm o
        mesmo município, a "distância" é 2). Isso se chama distância de Hamming.
      - em vez de MÉDIA, usa a MODA (o valor categórico mais frequente do grupo)
        como centro. Daí o nome "modes".

K-PROTOTYPES (para dados MISTOS — o nosso caso)
    Seus dados têm AMBOS: numéricos (idade) e categóricos (sexo, especialidade,
    município, classificação, desfecho). O k-prototypes combina os dois mundos:
      - nas colunas NUMÉRICAS, usa distância ao estilo k-means;
      - nas colunas CATEGÓRICAS, usa a contagem de diferenças ao estilo k-modes;
      - soma as duas (com um peso 'gamma' que equilibra a importância de cada
        tipo) para obter uma distância única entre as linhas.
    Por isso, mais abaixo, o script precisa informar QUAIS colunas são
    categóricas (a lista 'indices_categoricos'): é assim que o k-prototypes
    sabe onde aplicar cada tipo de cálculo.

RESUMO
    dados só numéricos ......... k-means
    dados só categóricos ....... k-modes
    dados mistos (nosso caso) .. k-prototypes  <-- usado aqui
-----------------------------------------------------------------------------
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sqlalchemy import create_engine
from data_profiling import ProfileReport
from kmodes.kprototypes import KPrototypes

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# Interruptores: escolha quais etapas rodar
RODAR_EXPLORACAO = True    # gera relatorio.html
RODAR_CLUSTERING = True    # descobre classes e gera clusters_resultado.csv

# --- Clustering ---
# Coluna de DESFECHO: separada do clustering, comparada com os grupos no final
COLUNA_ALVO = "outcome_class"

# Colunas REDUNDANTES a remover antes de clusterizar (comente p/ manter e comparar)
COLUNAS_REDUNDANTES = [
    "birth_year",     # idêntica a 'age' (corr ~ -1); mantemos 'age'
    "outcome_text",   # idêntica a 'outcome_class'
    # Geográficas: descomente para remover as menos granulares e manter cep_prefix
    # "municipality",
    # "state_code",
]

# Identificadores: nunca entram no clustering (são só chaves)
COLUNAS_ID = ["id", "patient_id", "attendance_id", "clinic_id"]

K_MIN = 2                 # menor número de clusters a testar
K_MAX = 6                 # maior número de clusters a testar
K_ESCOLHIDO = 6           # k a analisar em detalhe (ajuste após ver o "cotovelo")

# --- Tamanho das amostras (a query completa tem ~4,1 milhões de linhas) ---
# Amostramos DIRETO NO BANCO (ORDER BY random() LIMIT) para não trafegar nem
# guardar tudo na memória. Uma amostra aleatória representa bem a estrutura dos
# dados para explorar e achar classes — não é preciso as 4M de linhas.
#
# Limites pensados para uma máquina com 16 GB de RAM:
#   - Exploração (data-profiling) é GULOSA de memória: mantenha <= 100 mil.
#   - Clustering (k-prototypes) é mais leve: pode usar bem mais.
# O script baixa do banco a MAIOR das duas amostras e a exploração usa um
# subconjunto dela (assim conecta ao banco uma única vez).
LIMITE_CLUSTERING = 200000   # amostra para a Etapa 2 (folgado p/ 16 GB)
LIMITE_EXPLORACAO = 75000    # amostra para a Etapa 1 (seguro p/ 16 GB)

# =============================================================================
# CARGA DOS DADOS (compartilhada pelas duas etapas)
# =============================================================================

engine = create_engine("postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl")

# Baixamos do banco a MAIOR das duas amostras necessárias.
limite_carga = max(LIMITE_CLUSTERING, LIMITE_EXPLORACAO)

# ORDER BY random() faz o Postgres sortear as linhas antes de aplicar o LIMIT,
# entregando uma amostra ALEATÓRIA (e não só "as primeiras N linhas").
# Obs: em 4M de linhas esse random() leva algum tempo (o banco embaralha tudo);
# é normal a query demorar mais que um SELECT simples.
query = f"""
    SELECT
        a.*,
        p.*,
        co.*,
        e.analyte,
        e.classification
    FROM clinical.attendances          a
    JOIN clinical.patients             p  ON p.patient_id     = a.patient_id
    JOIN metrics.clinical_outcomes     co ON co.attendance_id = a.attendance_id
    JOIN metrics.exam_records          e  ON e.attendance_id  = a.attendance_id
    WHERE e.analyte IS NOT NULL
      AND e.classification IS NOT NULL
      AND (co.outcome_at - a.attended_at) >= 0
    ORDER BY random()
    LIMIT {limite_carga}
"""

print(f"Carregando amostra de até {limite_carga:,} linhas do banco...")
print("  (a base completa tem ~4,1 milhões; o ORDER BY random() pode demorar)")
df = pd.read_sql(query, engine)

# Renomeia colunas duplicadas: attendance_id, attendance_id.1, ...
cols = pd.Series(df.columns)
for dup in cols[cols.duplicated()].unique():
    idx = cols[cols == dup].index
    cols[idx] = [dup if i == 0 else f"{dup}.{i}" for i in range(len(idx))]
df.columns = cols

print(f"  -> {len(df):,} linhas, {len(df.columns)} colunas carregadas\n")

# =============================================================================
# ETAPA 1 — EXPLORAÇÃO (relatório visual)
# =============================================================================

if RODAR_EXPLORACAO:
    print("=" * 60)
    print("ETAPA 1 — Gerando relatório de exploração...")
    print("=" * 60)
    # Usa um subconjunto menor (LIMITE_EXPLORACAO), pois o data-profiling é
    # guloso de memória. Se a carga já veio menor que esse limite, usa tudo.
    if len(df) > LIMITE_EXPLORACAO:
        df_exp = df.sample(n=LIMITE_EXPLORACAO, random_state=42)
        print(f"  Usando {len(df_exp):,} linhas para a exploração")
    else:
        df_exp = df
    ProfileReport(df_exp, title="Relatório MOSAIC").to_file("relatorio.html")
    print("Relatório gerado: relatorio.html")
    print("  Abra com: xdg-open relatorio.html\n")

# =============================================================================
# ETAPA 2 — CLUSTERING (descoberta de classes)
# =============================================================================

if RODAR_CLUSTERING:
    print("=" * 60)
    print("ETAPA 2 — Descobrindo classes (k-prototypes)...")
    print("=" * 60)

    # Separa o alvo (desfecho) — não entra no clustering
    alvo = None
    if COLUNA_ALVO in df.columns:
        alvo = df[COLUNA_ALVO].copy()
        print(f"  -> Coluna-alvo '{COLUNA_ALVO}' separada para análise posterior")
    else:
        print(f"  ! Aviso: coluna-alvo '{COLUNA_ALVO}' não encontrada; seguindo sem alvo")

    # Monta a lista de colunas a descartar do clustering
    descartar = set(COLUNAS_REDUNDANTES) | set(COLUNAS_ID) | {COLUNA_ALVO}
    for c in df.columns:
        if c.split(".")[0] in COLUNAS_ID:   # pega duplicatas renomeadas de IDs
            descartar.add(c)

    X = df.drop(columns=[c for c in descartar if c in df.columns], errors="ignore")
    print(f"  -> {X.shape[1]} colunas no clustering: {list(X.columns)}")

    # Amostragem para o clustering (o df já veio amostrado do banco, mas se
    # LIMITE_CLUSTERING for menor que a carga, reduz mais aqui)
    if LIMITE_CLUSTERING and len(X) > LIMITE_CLUSTERING:
        X = X.sample(n=LIMITE_CLUSTERING, random_state=42)
        if alvo is not None:
            alvo = alvo.loc[X.index]
        print(f"  -> Usando amostra de {len(X):,} linhas")

    # Trata faltantes: categóricos -> 'DESCONHECIDO'; numéricos -> mediana
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype).startswith("category"):
            X[col] = X[col].astype(str).fillna("DESCONHECIDO")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())

    # Índices das colunas categóricas.
    # É AQUI que o conceito explicado no topo vira prática: o k-prototypes
    # precisa saber quais colunas são categóricas para aplicar a distância de
    # Hamming (contar diferenças) nelas e a distância estilo k-means nas
    # numéricas. Sem essa lista, ele trataria tudo como numérico e erraria.
    indices_categoricos = [
        i for i, col in enumerate(X.columns)
        if X[col].dtype == object or str(X[col].dtype).startswith("category")
    ]
    print(f"  -> {len(indices_categoricos)} categóricas, "
          f"{X.shape[1] - len(indices_categoricos)} numéricas\n")

    matriz = X.to_numpy()

    # Testa vários k e mostra o custo (procure o "cotovelo")
    print("Testando diferentes números de clusters (k):")
    print(f"{'k':>3} | {'custo (cost)':>15}")
    print("-" * 25)
    resultados = {}
    for k in range(K_MIN, K_MAX + 1):
        kproto = KPrototypes(n_clusters=k, init="Huang", random_state=42,
                             n_init=2, verbose=0)
        labels = kproto.fit_predict(matriz, categorical=indices_categoricos)
        resultados[k] = {"labels": labels, "custo": kproto.cost_}
        print(f"{k:>3} | {kproto.cost_:>15,.0f}")

    # Analisa a solução escolhida
    melhor = resultados[K_ESCOLHIDO]
    X_result = X.copy()
    X_result["cluster"] = melhor["labels"]
    if alvo is not None:
        X_result[COLUNA_ALVO] = alvo.values

    print(f"\n=== Solução com k = {K_ESCOLHIDO} ===")
    print("\nTamanho de cada cluster:")
    print(X_result["cluster"].value_counts().sort_index().to_string())

    # Cruzamento com o desfecho — o achado mais importante
    if alvo is not None:
        print(f"\nDistribuição de '{COLUNA_ALVO}' em cada cluster (%):")
        tabela = pd.crosstab(X_result["cluster"], X_result[COLUNA_ALVO],
                             normalize="index") * 100
        print(tabela.round(1).to_string())
        print("\n>>> Clusters que CONCENTRAM um desfecho são as classes relevantes.")

    # Perfil de cada cluster (categoria mais frequente por coluna)
    print("\nPerfil de cada cluster (valor mais comum por coluna):")
    for cl in sorted(X_result["cluster"].unique()):
        sub = X_result[X_result["cluster"] == cl]
        print(f"\n  Cluster {cl} ({len(sub)} pacientes):")
        for col in X.columns:
            if X[col].dtype == object or str(X[col].dtype).startswith("category"):
                top = sub[col].mode()
                if len(top) > 0:
                    print(f"    {col}: {top.iloc[0]}")

    # Salva resultado
    saida = "clusters_resultado.csv"
    X_result.to_csv(saida, index=False)
    print(f"\nResultado salvo em: {saida}")
    print("  (cada linha com seu cluster — use para gráficos ou no Superset)")

print("\nPipeline concluída.")
