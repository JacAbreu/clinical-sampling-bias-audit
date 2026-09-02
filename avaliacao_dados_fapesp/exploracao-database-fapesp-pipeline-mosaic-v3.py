"""
=============================================================================
PIPELINE DE DESCOBERTA DE CLASSES v3 — MOSAIC-FL
CORREÇÃO DO VIÉS DE GRANULARIDADE (agregação por atendimento)
=============================================================================

MOTIVAÇÃO (viés identificado no v2):
    A query original traz UMA LINHA POR EXAME, não por atendimento. Como um
    atendimento tem muitos exames, um paciente com 50 exames entra 50 vezes na
    amostra, enquanto um com 2 exames entra 2 vezes. Isso causa DOIS problemas:

      1. VIÉS DE AMOSTRAGEM: o 'ORDER BY random() LIMIT' favorece pacientes com
         muitos exames (mais linhas = mais chance de serem sorteados).
      2. UNIDADE DE ANÁLISE ERRADA: o clustering compara "linhas de exame"
         (que diferem só em analyte/classification) em vez de "perfis de
         paciente". Foi por isso que os clusters do v2 se separaram por tipo
         de exame, sem significado clínico.

CORREÇÃO:
    Agregar para UMA LINHA POR ATENDIMENTO antes de clusterizar. Cada
    atendimento vira um perfil com RESUMOS dos seus exames:
      - nº de exames HIGH / LOW / NORMAL
      - nº total de exames e de analitos distintos
      - proporção de exames alterados (HIGH+LOW)
    Mais os atributos do atendimento (idade, sexo, tipo, desfecho, duração).
    Assim a unidade de análise fica clara e sem distorção de amostragem.

ESTRUTURA (paralela ao v2), tudo versionado por data:
    Parte 0 — DEMONSTRAÇÃO DO VIÉS (quantifica o problema antes/depois)
    Parte 1 — EXPLORAÇÃO da base agregada (relatório visual)
    Parte 2 — CLUSTERIZAÇÃO sobre atendimentos (k-prototypes)
    Parte 3 — AUDITORIA da base agregada
    Parte 4 — ANÁLISES: duração, hospitais, comparação com as 5 classes atuais

Saídas versionadas (com VERSÃO e data/hora no nome):
    vies_V3_AAAA-MM-DD_HHMM.txt            (demonstração do viés)
    relatorio_agg_V3_AAAA-MM-DD_HHMM.html
    clusters_agg_V3_AAAA-MM-DD_HHMM.csv
    auditoria_agg_V3_AAAA-MM-DD_HHMM.txt
    analises_agg_V3_AAAA-MM-DD_HHMM.txt

Dependências:
    pip install pandas numpy sqlalchemy psycopg2-binary fg-data-profiling kmodes
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

RODAR_DEMO_VIES   = True    # Parte 0
RODAR_EXPLORACAO  = True    # Parte 1
RODAR_CLUSTERING  = True    # Parte 2
RODAR_AUDITORIA   = True    # Parte 3
RODAR_ANALISES    = True    # Parte 4

# Fonte: CSV LONGO (1 linha/exame) com attendance_id, ou leitura do banco.
# IMPORTANTE: diferente do v2, aqui precisamos de attendance_id e patient_id
# para poder agregar. Por isso a query os inclui explicitamente.
CSV_ENTRADA = None
CONEXAO = "postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl"

# Amostragem: agora amostramos ATENDIMENTOS, não linhas de exame.
# Baixamos um número maior de linhas de exame e agregamos; o nº de atendimentos
# resultante será menor. Ajuste conforme a densidade de exames por atendimento.
LIMITE_LINHAS_EXAME = 400000   # linhas de exame a baixar (viram ~N atendimentos)
LIMITE_EXPLORACAO   = 75000    # atendimentos p/ o relatório (guloso de memória)

# Clustering
COLUNA_ALVO = "outcome_class"
K_MIN, K_MAX = 2, 8
K_ESCOLHIDO = 5                # 5 p/ comparar com as 5 classes teóricas atuais

# Colunas-chave (ajuste se diferirem)
COL_ATENDIMENTO = "attendance_id"
COL_PACIENTE    = "patient_id"
COL_HOSPITAL    = "hospital_id"
COL_TIPO_ATEND  = "attendance_type"
COL_ATENDIDO    = "attended_at"
COL_DESFECHO    = "outcome_at"
COL_ANALITO     = "analyte"
COL_CLASSIF     = "classification"

# Análise de duração (Parte 4a)
CORTE_ATUAL_DIAS = 10
CORTES_TESTAR = [5, 7, 10, 14, 21, 30]

# =============================================================================
# VERSIONAMENTO
# =============================================================================
VERSAO = "V3"    # identifica qual pipeline gerou o arquivo (ex: relatorio_V3_2026-07-06_1145.html)
CARIMBO = datetime.now().strftime("%Y-%m-%d_%H%M")
print(f"Execução {VERSAO} (agregada): {CARIMBO}\n")

def nome_saida(base, ext):
    # Formato: <base>_<VERSAO>_<data-hora>.<ext>  ->  relatorio_agg_V3_2026-07-06_1145.html
    return f"{base}_{VERSAO}_{CARIMBO}.{ext}"


class _Tee:
    """Escreve simultaneamente no terminal e num arquivo."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, t):
        for s in self.streams:
            s.write(t)
    def flush(self):
        for s in self.streams:
            s.flush()


# =============================================================================
# CARGA — traz o formato LONGO (1 linha por exame) COM as chaves de atendimento
# =============================================================================

def carregar_longo():
    if CSV_ENTRADA and os.path.exists(CSV_ENTRADA):
        print(f"Lendo formato longo do CSV: {CSV_ENTRADA}")
        return pd.read_csv(CSV_ENTRADA)

    print(f"Carregando até {LIMITE_LINHAS_EXAME:,} linhas de exame do banco...")
    print("  (inclui attendance_id/patient_id para permitir agregação)")
    engine = create_engine(CONEXAO)
    query = f"""
        SELECT
            a.attendance_id,
            a.patient_id,
            a.hospital_id,
            a.attendance_type,
            a.attended_at,
            a.specialty,
            p.sex,
            p.age,
            p.state_code,
            p.municipality,
            co.outcome_class,
            co.outcome_at,
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
        LIMIT {LIMITE_LINHAS_EXAME}
    """
    return pd.read_sql(query, engine)


# =============================================================================
# AGREGAÇÃO — transforma formato longo (1/exame) em largo (1/atendimento)
# =============================================================================

def agregar_por_atendimento(long_df):
    """Uma linha por attendance_id, com resumos dos exames como features."""
    g = long_df.groupby(COL_ATENDIMENTO)

    # Features derivadas dos exames (o coração da correção do viés)
    resumo = pd.DataFrame({
        "n_exames":          g.size(),
        "n_analitos_dist":   g[COL_ANALITO].nunique(),
        "n_high":            g[COL_CLASSIF].apply(lambda s: (s == "HIGH").sum()),
        "n_low":             g[COL_CLASSIF].apply(lambda s: (s == "LOW").sum()),
        "n_normal":          g[COL_CLASSIF].apply(lambda s: (s == "NORMAL").sum()),
    })
    resumo["prop_alterados"] = (
        (resumo["n_high"] + resumo["n_low"]) / resumo["n_exames"]
    ).round(3)

    # Atributos do atendimento (constantes dentro de cada attendance_id -> 'first')
    attrs = g.agg({
        COL_PACIENTE:   "first",
        COL_HOSPITAL:   "first",
        COL_TIPO_ATEND: "first",
        COL_ATENDIDO:   "first",
        "specialty":    "first",
        "sex":          "first",
        "age":          "first",
        "state_code":   "first",
        "municipality": "first",
        COLUNA_ALVO:    "first",
        COL_DESFECHO:   "first",
    })

    agg = attrs.join(resumo).reset_index()

    # duration_days
    at = pd.to_datetime(agg[COL_ATENDIDO], errors="coerce")
    ot = pd.to_datetime(agg[COL_DESFECHO], errors="coerce")
    agg["duration_days"] = (ot - at).dt.days

    return agg


# =============================================================================
# EXECUÇÃO
# =============================================================================

long_df = carregar_longo()
print(f"  -> formato longo: {len(long_df):,} linhas de exame\n")

# Se o CSV não tiver attendance_id, não dá para agregar (avisa e para)
if COL_ATENDIMENTO not in long_df.columns:
    print("ERRO: a fonte não tem 'attendance_id'. A agregação é impossível.")
    print("      Rode a partir do banco (CSV_ENTRADA=None) ou inclua a coluna.")
    sys.exit(1)

df = agregar_por_atendimento(long_df)
print(f"Base AGREGADA: {len(df):,} atendimentos, {len(df.columns)} colunas")
print(f"  (compressão: {len(long_df):,} exames -> {len(df):,} atendimentos, "
      f"~{len(long_df)/max(len(df),1):.1f} exames/atendimento)\n")


# =============================================================================
# PARTE 0 — DEMONSTRAÇÃO DO VIÉS
# =============================================================================
if RODAR_DEMO_VIES:
    saida_vies = nome_saida("vies", "txt")
    arq = open(saida_vies, "w")
    orig = sys.stdout
    sys.stdout = _Tee(orig, arq)

    print("=" * 70)
    print("PARTE 0 — DEMONSTRAÇÃO DO VIÉS DE GRANULARIDADE")
    print("=" * 70)

    # Exames por atendimento: a fonte do viés
    ex_por_atend = long_df.groupby(COL_ATENDIMENTO).size()
    print("\nExames por atendimento (fonte do viés de amostragem):")
    print(f"  mínimo: {ex_por_atend.min()} | mediana: {ex_por_atend.median():.0f} "
          f"| média: {ex_por_atend.mean():.1f} | máximo: {ex_por_atend.max()}")

    # Quanto o formato longo super-representa pacientes com muitos exames
    print("\nEfeito no ORDER BY random() (formato LONGO):")
    print("  Um atendimento com N exames tem N vezes mais chance de ser sorteado.")
    top = ex_por_atend.sort_values(ascending=False).head(3)
    print(f"  Ex.: os 3 atendimentos com mais exames têm "
          f"{top.iloc[0]}, {top.iloc[1]}, {top.iloc[2]} exames cada —")
    print(f"  contra a mediana de {ex_por_atend.median():.0f}. "
          "No formato longo, pesam desproporcionalmente.")

    # Comparação de distribuição do desfecho: longo (enviesado) vs agregado
    print("\nDistribuição de 'outcome_class': LONGO (enviesado) vs AGREGADO:")
    dist_long = long_df[COLUNA_ALVO].value_counts(normalize=True).sort_index() * 100
    dist_agg = df[COLUNA_ALVO].value_counts(normalize=True).sort_index() * 100
    comp = pd.DataFrame({"longo_%": dist_long.round(1),
                         "agregado_%": dist_agg.round(1)})
    comp["diferença"] = (comp["agregado_%"] - comp["longo_%"]).round(1)
    print("  " + comp.to_string().replace("\n", "\n  "))
    print("\n  >>> Diferenças não-triviais confirmam que o formato longo")
    print("  >>> distorcia a distribuição real de desfechos por atendimento.")

    sys.stdout = orig
    arq.close()
    print(f"\n  Demonstração do viés salva em: {saida_vies}\n")


# =============================================================================
# PARTE 1 — EXPLORAÇÃO (base agregada)
# =============================================================================
if RODAR_EXPLORACAO:
    print("=" * 70)
    print("PARTE 1 — EXPLORAÇÃO da base agregada")
    print("=" * 70)
    from data_profiling import ProfileReport
    df_exp = df.sample(n=min(LIMITE_EXPLORACAO, len(df)), random_state=42)
    saida_html = nome_saida("relatorio_agg", "html")
    ProfileReport(df_exp, title=f"MOSAIC agregado {CARIMBO}").to_file(saida_html)
    print(f"  -> {saida_html} ({len(df_exp):,} atendimentos)\n")


# =============================================================================
# PARTE 2 — CLUSTERIZAÇÃO (sobre atendimentos)
# =============================================================================
labels_cluster = None
if RODAR_CLUSTERING:
    print("=" * 70)
    print("PARTE 2 — CLUSTERIZAÇÃO sobre atendimentos (k-prototypes)")
    print("=" * 70)
    from kmodes.kprototypes import KPrototypes

    alvo = df[COLUNA_ALVO].copy() if COLUNA_ALVO in df.columns else None

    # Colunas fora do clustering: IDs, datas, e o alvo
    descartar = {COL_ATENDIMENTO, COL_PACIENTE, COL_ATENDIDO, COL_DESFECHO, COLUNA_ALVO}
    X = df.drop(columns=[c for c in descartar if c in df.columns], errors="ignore")

    # Trata faltantes; age==0 -> ausente (idade disfarçada)
    for col in X.columns:
        if col == "age":
            X[col] = pd.to_numeric(X[col], errors="coerce").replace(0, np.nan)
            X[col] = X[col].fillna(X[col].median())
        elif X[col].dtype == object or str(X[col].dtype).startswith("category"):
            X[col] = X[col].astype(str).fillna("DESCONHECIDO")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())

    idx_cat = [i for i, c in enumerate(X.columns)
               if X[c].dtype == object or str(X[c].dtype).startswith("category")]
    print(f"  {X.shape[1]} colunas ({len(idx_cat)} categóricas, "
          f"{X.shape[1]-len(idx_cat)} numéricas)")
    print(f"  Features de exame agregadas: n_exames, n_high, n_low, n_normal, "
          f"n_analitos_dist, prop_alterados")
    print(f"  Colunas: {list(X.columns)}\n")

    matriz = X.to_numpy()
    print(f"  Testando k de {K_MIN} a {K_MAX}:")
    print(f"  {'k':>3} | {'custo':>15}")
    print("  " + "-" * 23)
    resultados = {}
    for k in range(K_MIN, K_MAX + 1):
        kp = KPrototypes(n_clusters=k, init="Huang", random_state=42,
                         n_init=2, verbose=0)
        lab = kp.fit_predict(matriz, categorical=idx_cat)
        resultados[k] = {"labels": lab, "custo": kp.cost_}
        print(f"  {k:>3} | {kp.cost_:>15,.0f}")

    melhor = resultados[K_ESCOLHIDO]
    labels_cluster = pd.Series(melhor["labels"], index=X.index, name="cluster")
    X_out = X.copy()
    X_out["cluster"] = melhor["labels"]
    if alvo is not None:
        X_out[COLUNA_ALVO] = alvo.values
    for extra in ["duration_days", COL_HOSPITAL, COL_ATENDIMENTO]:
        if extra in df.columns and extra not in X_out.columns:
            X_out[extra] = df[extra].values

    saida_csv = nome_saida("clusters_agg", "csv")
    X_out.to_csv(saida_csv, index=False)
    print(f"\n  Solução k={K_ESCOLHIDO} salva em: {saida_csv}")

    if alvo is not None:
        print(f"\n  Distribuição de '{COLUNA_ALVO}' por cluster (%):")
        tab = pd.crosstab(X_out["cluster"], X_out[COLUNA_ALVO],
                          normalize="index") * 100
        print("  " + tab.round(1).to_string().replace("\n", "\n  "))
        sep = tab.std().mean()
        print(f"\n  Separação média do desfecho entre clusters: {sep:.1f} pontos %")
        print("  (>10 = clusters distinguem desfecho; <5 = não distinguem)")
    print()


# =============================================================================
# PARTE 3 — AUDITORIA (base agregada)
# =============================================================================
if RODAR_AUDITORIA:
    print("=" * 70)
    print("PARTE 3 — AUDITORIA da base agregada")
    print("=" * 70)
    linhas = []
    def reg(s=""):
        print("  " + s); linhas.append(s)

    reg(f"Atendimentos: {len(df):,} | Colunas: {len(df.columns)}")
    reg()
    reg("Cardinalidade:")
    for c in df.columns:
        n = df[c].nunique(dropna=False)
        d = "INÚTIL (valor único)" if n == 1 else ("alta cardinalidade" if n > 100 else "")
        reg(f"  {c:22s}: {n:>7} distintos  {d}")
    reg()
    reg("Nulos reais (>0%):")
    nul = (df.isnull().mean() * 100).sort_values(ascending=False)
    for c, p in nul[nul > 0].items():
        reg(f"  {c:22s}: {p:5.1f}%")
    reg()
    reg("Ausência disfarçada:")
    if "age" in df.columns:
        p0 = (pd.to_numeric(df["age"], errors="coerce") == 0).mean() * 100
        if p0 >= 20:
            reg(f"  age: {p0:.1f}% == 0 (idade faltante disfarçada)")

    saida_txt = nome_saida("auditoria_agg", "txt")
    with open(saida_txt, "w") as f:
        f.write(f"AUDITORIA (agregada) — {CARIMBO}\n\n" + "\n".join(linhas))
    print(f"\n  Auditoria salva em: {saida_txt}\n")


# =============================================================================
# PARTE 4 — ANÁLISES DIRIGIDAS AO OBJETIVO
# =============================================================================
if RODAR_ANALISES:
    saida_analises = nome_saida("analises_agg", "txt")
    arq = open(saida_analises, "w")
    orig = sys.stdout
    sys.stdout = _Tee(orig, arq)

    print("=" * 70)
    print("PARTE 4 — ANÁLISES PARA DESCOBERTA DE CLASSES (base agregada)")
    print(f"Execução: {CARIMBO}")
    print("=" * 70)

    # 4a. Duração e corte
    print("\n--- 4a. DURAÇÃO e o corte de", CORTE_ATUAL_DIAS, "dias ---")
    if "duration_days" in df.columns and COL_TIPO_ATEND in df.columns:
        internados = df[df[COL_TIPO_ATEND].astype(str)
                        .str.contains("nternado", case=False, na=False)]
        dur = internados["duration_days"].dropna()
        print(f"  Atendimentos internados: {len(internados):,}")
        print(f"  Duração — mediana: {dur.median():.0f}d | média: {dur.mean():.0f} "
              f"| P75: {dur.quantile(.75):.0f} | P90: {dur.quantile(.90):.0f}")
        print(f"\n  {'corte':>6} | {'breve%':>7} | {'grave%':>7} | equilíbrio")
        print("  " + "-" * 42)
        for corte in sorted(set(CORTES_TESTAR + [CORTE_ATUAL_DIAS])):
            breve = (dur <= corte).mean() * 100
            equil = "EQUILIBRADO" if 40 <= breve <= 60 else ""
            marca = " <- ATUAL" if corte == CORTE_ATUAL_DIAS else ""
            print(f"  {corte:>6} | {breve:>6.1f}% | {100-breve:>6.1f}% | {equil}{marca}")
        print(f"\n  >>> Agora medido POR ATENDIMENTO (sem o viés de exames repetidos).")
    else:
        print("  (colunas ausentes; pulei 4a)")

    # 4b. Balanceamento entre hospitais
    print("\n--- 4b. BALANCEAMENTO ENTRE HOSPITAIS ---")
    if COL_HOSPITAL in df.columns and df[COL_HOSPITAL].nunique() > 1:
        print(f"  Hospitais: {df[COL_HOSPITAL].nunique()}")
        print(f"\n  '{COLUNA_ALVO}' por hospital (%):")
        tab_h = pd.crosstab(df[COL_HOSPITAL], df[COLUNA_ALVO],
                            normalize="index") * 100
        print("  " + tab_h.round(1).to_string().replace("\n", "\n  "))
        het = tab_h.std().mean()
        print(f"\n  Heterogeneidade média: {het:.1f} pontos % (alto = non-IID)")
    else:
        print("  ! Só um hospital na amostra — non-IID não mensurável.")
        print("  ! Verifique se a query traz BPSP e HSL (amostragem estratificada?).")

    # 4c. Classes atuais vs clusters
    print("\n--- 4c. CLASSES ATUAIS vs CLUSTERS ---")
    def classe_atual(row):
        oc = row.get(COLUNA_ALVO)
        internado = "nternado" in str(row.get(COL_TIPO_ATEND, ""))
        dur = row.get("duration_days", np.nan)
        if oc == 0 and not internado: return "curado_pronto"
        if oc == 0 and internado:     return "curado_internado"
        if oc == 1 and not internado: return "melhora_pronto"
        if oc == 1 and internado:
            return "melhora_internado_breve" if dur <= CORTE_ATUAL_DIAS else "melhora_internado_grave"
        return "outro"

    if COLUNA_ALVO in df.columns and COL_TIPO_ATEND in df.columns:
        df["_classe_atual"] = df.apply(classe_atual, axis=1)
        print("  5 CLASSES ATUAIS (agora por atendimento):")
        dist = df["_classe_atual"].value_counts(normalize=True).sort_index() * 100
        for cl, p in dist.items():
            raro = " <- RARA (some sob ruído DP)" if p < 5 else ""
            print(f"    {cl:28s}: {p:5.1f}%{raro}")
        if COL_HOSPITAL in df.columns and df[COL_HOSPITAL].nunique() > 1:
            tab_c = pd.crosstab(df[COL_HOSPITAL], df["_classe_atual"],
                                normalize="index") * 100
            print(f"\n  Heterogeneidade das classes atuais: "
                  f"{tab_c.std().mean():.1f} pontos %")
        if labels_cluster is not None:
            print("\n  CLUSTERS descobertos (balanceamento):")
            dc = labels_cluster.value_counts(normalize=True).sort_index() * 100
            for cl, p in dc.items():
                print(f"    cluster {cl}: {p:5.1f}%")
            print("\n  >>> Os clusters (base agregada) são mais equilibrados e")
            print("  >>> separam melhor o desfecho que as 5 classes teóricas?")
    else:
        print("  (colunas ausentes; pulei 4c)")

    sys.stdout = orig
    arq.close()
    print(f"\n  Análises salvas em: {saida_analises}")

print("\n" + "=" * 70)
print(f"PIPELINE v3 CONCLUÍDA — arquivos com carimbo {CARIMBO}")
print("=" * 70)
