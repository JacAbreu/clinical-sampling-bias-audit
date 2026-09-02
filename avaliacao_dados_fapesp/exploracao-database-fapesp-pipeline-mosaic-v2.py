"""
=============================================================================
PIPELINE DE DESCOBERTA DE CLASSES — MOSAIC-FL
=============================================================================
Objetivo: descobrir se existe um esquema de CLASSES, derivado dos DADOS, que
seja mais equilibrado entre hospitais e/ou separe melhor 'outcome_class' do
que as 5 classes teóricas atuais (curado_pronto, curado_internado,
melhora_pronto, melhora_internado_breve, melhora_internado_grave).

Contexto (ver docs do projeto):
  - As 5 classes atuais vieram de REGRA CLÍNICA, não de dados.
  - O limiar de 10 dias (breve vs grave) é fixo no código, nunca validado.
  - As classes são desbalanceadas/heterogêneas entre hospitais (BPSP vs HSL),
    o que prejudica a federação e some sob ruído de privacidade diferencial.

Cada execução gera arquivos VERSIONADOS por data/hora, para acumular
extrações e comparar visões ao longo do tempo:
    relatorio_AAAA-MM-DD_HHMM.html      (exploração)
    clusters_AAAA-MM-DD_HHMM.csv        (resultado do clustering)
    auditoria_AAAA-MM-DD_HHMM.txt       (qualidade dos dados)
    analises_AAAA-MM-DD_HHMM.txt        (duração, hospitais, comparação de classes)

ORDEM DAS PARTES:
    Parte 1 — EXPLORAÇÃO (relatório visual)
    Parte 2 — CLUSTERIZAÇÃO (k-prototypes)
    Parte 3 — AUDITORIA DE QUALIDADE
    Parte 4 — ANÁLISES DIRIGIDAS AO OBJETIVO:
        4a. Distribuição de duration_days e teste de cortes (valida os 10 dias)
        4b. Balanceamento das classes entre hospitais (non-IID)
        4c. Comparação: 5 classes atuais vs classes descobertas por cluster

Dependências:
    pip install pandas numpy sqlalchemy psycopg2-binary fg-data-profiling kmodes
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# --- Interruptores de etapa ---
RODAR_EXPLORACAO   = True
RODAR_CLUSTERING   = True
RODAR_AUDITORIA    = True
RODAR_ANALISES     = True   # Parte 4 (duração, hospitais, comparação)

# --- Fonte de dados ---
# Se um CSV existir, usa ele (rápido p/ iterar). Senão, baixa do banco.
CSV_ENTRADA = None          # ex: "dados_base.csv"; None força leitura do banco
CONEXAO = "postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl"

# --- Tamanho das amostras (base completa ~4,1 milhões de linhas) ---
LIMITE_CLUSTERING = 200000  # amostra p/ clustering (k-prototypes é leve)
LIMITE_EXPLORACAO = 75000   # amostra p/ data-profiling (guloso de memória; <=100k em 16GB)

# --- Clustering ---
COLUNA_ALVO = "outcome_class"
K_MIN, K_MAX = 2, 8
K_ESCOLHIDO = 5             # 5 p/ comparar diretamente com as 5 classes atuais

# Colunas que NÃO entram no clustering:
COLUNAS_ID = ["id", "patient_id", "attendance_id", "clinic_id", "hospital_id"]
# Removidas por não informarem (ajuste conforme a auditoria):
COLUNAS_REMOVER = [
    "suspected_diagnosis", "confirmed_diagnosis",  # 100% vazias
    "cep_prefix",                                  # ~100% vazia
    "attended_at", "outcome_at",                   # datas viram ruído
]

# --- Análise de duração (Parte 4a) ---
CORTE_ATUAL_DIAS = 10       # limiar atual (breve vs grave) a validar
CORTES_TESTAR = [5, 7, 10, 14, 21, 30]  # cortes alternativos a comparar

# --- Nomes de colunas-chave (ajuste se diferirem na sua base) ---
COL_HOSPITAL = "hospital_id"
COL_TIPO_ATEND = "attendance_type"
COL_ATENDIDO = "attended_at"
COL_DESFECHO = "outcome_at"

# =============================================================================
# VERSIONAMENTO — timestamp desta execução
# =============================================================================
CARIMBO = datetime.now().strftime("%Y-%m-%d_%H%M")
print(f"Execução: {CARIMBO}\n")

def nome_saida(base, ext):
    return f"{base}_{CARIMBO}.{ext}"

# =============================================================================
# CARGA DOS DADOS (compartilhada por todas as partes)
# =============================================================================

def carregar():
    if CSV_ENTRADA and os.path.exists(CSV_ENTRADA):
        print(f"Lendo dados do CSV: {CSV_ENTRADA}")
        return pd.read_csv(CSV_ENTRADA)

    print(f"Carregando amostra de até {LIMITE_CLUSTERING:,} linhas do banco...")
    print("  (base completa ~4,1 milhões; ORDER BY random() pode demorar)")
    engine = create_engine(CONEXAO)
    limite = max(LIMITE_CLUSTERING, LIMITE_EXPLORACAO)
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
        LIMIT {limite}
    """
    df = pd.read_sql(query, engine)
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        idx = cols[cols == dup].index
        cols[idx] = [dup if i == 0 else f"{dup}.{i}" for i in range(len(idx))]
    df.columns = cols
    return df


df = carregar()
print(f"  -> {len(df):,} linhas, {len(df.columns)} colunas\n")

# Deriva duration_days (necessária para várias análises)
if COL_ATENDIDO in df.columns and COL_DESFECHO in df.columns:
    at = pd.to_datetime(df[COL_ATENDIDO], errors="coerce")
    ot = pd.to_datetime(df[COL_DESFECHO], errors="coerce")
    df["duration_days"] = (ot - at).dt.days
    print(f"Coluna 'duration_days' derivada (mediana: "
          f"{df['duration_days'].median():.0f} dias)\n")

# =============================================================================
# PARTE 1 — EXPLORAÇÃO (relatório visual)
# =============================================================================
if RODAR_EXPLORACAO:
    print("=" * 70)
    print("PARTE 1 — EXPLORAÇÃO (relatório visual)")
    print("=" * 70)
    from data_profiling import ProfileReport
    df_exp = df.sample(n=min(LIMITE_EXPLORACAO, len(df)), random_state=42)
    saida_html = nome_saida("relatorio", "html")
    ProfileReport(df_exp, title=f"Relatório MOSAIC {CARIMBO}").to_file(saida_html)
    print(f"  -> {saida_html} ({len(df_exp):,} linhas)\n")

# =============================================================================
# PARTE 2 — CLUSTERIZAÇÃO (k-prototypes)
# =============================================================================
labels_cluster = None
if RODAR_CLUSTERING:
    print("=" * 70)
    print("PARTE 2 — CLUSTERIZAÇÃO (k-prototypes)")
    print("=" * 70)
    from kmodes.kprototypes import KPrototypes

    alvo = df[COLUNA_ALVO].copy() if COLUNA_ALVO in df.columns else None

    descartar = set(COLUNAS_ID) | set(COLUNAS_REMOVER) | {COLUNA_ALVO}
    for c in df.columns:
        if c.split(".")[0] in COLUNAS_ID:
            descartar.add(c)
    X = df.drop(columns=[c for c in descartar if c in df.columns], errors="ignore")

    if LIMITE_CLUSTERING and len(X) > LIMITE_CLUSTERING:
        X = X.sample(n=LIMITE_CLUSTERING, random_state=42)
        if alvo is not None:
            alvo = alvo.loc[X.index]

    # Trata faltantes; age==0 tratado como ausente (idade disfarçada)
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
    print(f"  Colunas: {list(X.columns)}\n")

    matriz = X.to_numpy()
    print(f"  Testando k de {K_MIN} a {K_MAX}:")
    print(f"  {'k':>3} | {'custo':>15}")
    print("  " + "-" * 23)
    resultados = {}
    for k in range(K_MIN, K_MAX + 1):
        kp = KPrototypes(n_clusters=k, init="Huang", random_state=42, n_init=2, verbose=0)
        lab = kp.fit_predict(matriz, categorical=idx_cat)
        resultados[k] = {"labels": lab, "custo": kp.cost_}
        print(f"  {k:>3} | {kp.cost_:>15,.0f}")

    melhor = resultados[K_ESCOLHIDO]
    labels_cluster = pd.Series(melhor["labels"], index=X.index, name="cluster")
    X_out = X.copy()
    X_out["cluster"] = melhor["labels"]
    if alvo is not None:
        X_out[COLUNA_ALVO] = alvo.values
    if "duration_days" in df.columns:
        X_out["duration_days"] = df.loc[X.index, "duration_days"].values
    if COL_HOSPITAL in df.columns:
        X_out[COL_HOSPITAL] = df.loc[X.index, COL_HOSPITAL].values

    saida_csv = nome_saida("clusters", "csv")
    X_out.to_csv(saida_csv, index=False)
    print(f"\n  Solução k={K_ESCOLHIDO} salva em: {saida_csv}")

    if alvo is not None:
        print(f"\n  Distribuição de '{COLUNA_ALVO}' por cluster (%):")
        tab = pd.crosstab(X_out["cluster"], X_out[COLUNA_ALVO], normalize="index") * 100
        print("  " + tab.round(1).to_string().replace("\n", "\n  "))
        # Qualidade da separação: quanto maior o desvio entre clusters, melhor
        sep = tab.std().mean()
        print(f"\n  Separação média do desfecho entre clusters: {sep:.1f} pontos %")
        print("  (>10 = clusters distinguem desfecho; <5 = não distinguem)")
    print()

# =============================================================================
# PARTE 3 — AUDITORIA DE QUALIDADE
# =============================================================================
if RODAR_AUDITORIA:
    print("=" * 70)
    print("PARTE 3 — AUDITORIA DE QUALIDADE")
    print("=" * 70)
    linhas_audit = []
    def reg(s=""):
        print("  " + s)
        linhas_audit.append(s)

    reg(f"Linhas: {len(df):,} | Colunas: {len(df.columns)}")
    reg()
    reg("Cardinalidade e diagnóstico:")
    inuteis = []
    for c in df.columns:
        n = df[c].nunique(dropna=False)
        d = ""
        if n == 1:
            d = "INÚTIL (valor único)"; inuteis.append(c)
        elif n > 100:
            d = "alta cardinalidade"
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

    saida_txt = nome_saida("auditoria", "txt")
    with open(saida_txt, "w") as f:
        f.write(f"AUDITORIA DE QUALIDADE — {CARIMBO}\n\n" + "\n".join(linhas_audit))
    print(f"\n  Auditoria salva em: {saida_txt}\n")

# =============================================================================
# PARTE 4 — ANÁLISES DIRIGIDAS AO OBJETIVO
# =============================================================================
if RODAR_ANALISES:
    import sys

    # "Tee": tudo que for impresso na Parte 4 vai para o terminal E para um
    # arquivo versionado por data, permitindo acumular análises entre execuções.
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, texto):
            for s in self.streams:
                s.write(texto)
        def flush(self):
            for s in self.streams:
                s.flush()

    saida_analises = nome_saida("analises", "txt")
    _arquivo_analises = open(saida_analises, "w")
    _stdout_original = sys.stdout
    sys.stdout = _Tee(_stdout_original, _arquivo_analises)

    print("=" * 70)
    print("PARTE 4 — ANÁLISES PARA DESCOBERTA DE CLASSES")
    print(f"Execução: {CARIMBO}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 4a. Distribuição de duração e validação do corte de 10 dias
    # -------------------------------------------------------------------------
    print("\n--- 4a. DURAÇÃO DE INTERNAÇÃO e o corte de", CORTE_ATUAL_DIAS, "dias ---")
    if "duration_days" in df.columns and COL_TIPO_ATEND in df.columns:
        internados = df[df[COL_TIPO_ATEND].astype(str)
                        .str.contains("nternado", case=False, na=False)]
        dur = internados["duration_days"].dropna()
        print(f"  Internados: {len(internados):,}")
        print(f"  Duração — mediana: {dur.median():.0f} dias | "
              f"média: {dur.mean():.0f} | P75: {dur.quantile(.75):.0f} | "
              f"P90: {dur.quantile(.90):.0f}")
        print(f"\n  A regra atual usa corte fixo de {CORTE_ATUAL_DIAS} dias.")
        print("  Como cada corte dividiria os internados (breve% / grave%):")
        print(f"  {'corte':>6} | {'breve%':>7} | {'grave%':>7} | equilíbrio")
        print("  " + "-" * 42)
        for corte in sorted(set(CORTES_TESTAR + [CORTE_ATUAL_DIAS])):
            breve = (dur <= corte).mean() * 100
            grave = 100 - breve
            # equilíbrio: mais perto de 50/50 = melhor para balanceamento
            equil = "EQUILIBRADO" if 40 <= breve <= 60 else ""
            marca = " <- ATUAL" if corte == CORTE_ATUAL_DIAS else ""
            print(f"  {corte:>6} | {breve:>6.1f}% | {grave:>6.1f}% | {equil}{marca}")
        print(f"\n  >>> A mediana ({dur.median():.0f}) é o corte que gera "
              "divisão mais equilibrada (50/50).")
        print("  >>> Se o objetivo é balancear, considere o corte na mediana, "
              "não em 10 fixo.")
    else:
        print("  (duration_days ou attendance_type ausentes; pulei 4a)")

    # -------------------------------------------------------------------------
    # 4b. Balanceamento entre hospitais (non-IID)
    # -------------------------------------------------------------------------
    print("\n--- 4b. BALANCEAMENTO ENTRE HOSPITAIS ---")
    if COL_HOSPITAL in df.columns and df[COL_HOSPITAL].nunique() > 1:
        print(f"  Hospitais na amostra: {df[COL_HOSPITAL].nunique()}")
        print(f"\n  Distribuição de '{COLUNA_ALVO}' por hospital (%):")
        tab_h = pd.crosstab(df[COL_HOSPITAL], df[COLUNA_ALVO], normalize="index") * 100
        print("  " + tab_h.round(1).to_string().replace("\n", "\n  "))
        # Heterogeneidade: variação de cada classe entre hospitais
        het = tab_h.std().mean()
        print(f"\n  Heterogeneidade média entre hospitais: {het:.1f} pontos %")
        print("  (alto = non-IID severo, ruim p/ federação e privacidade)")
    else:
        print("  ! Só um hospital na amostra — não dá para medir non-IID.")
        print("  ! Para esta análise, garanta que a query traga BPSP e HSL.")
        print("  ! (verifique se o filtro/JOIN não está restringindo a um hospital)")

    # -------------------------------------------------------------------------
    # 4c. Comparação: 5 classes atuais vs classes por cluster
    # -------------------------------------------------------------------------
    print("\n--- 4c. CLASSES ATUAIS vs CLUSTERS DESCOBERTOS ---")

    # Reconstrói as 5 classes atuais pela regra documentada
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
        print("  Distribuição das 5 CLASSES ATUAIS (regra clínica):")
        dist = df["_classe_atual"].value_counts(normalize=True).sort_index() * 100
        for cl, p in dist.items():
            raro = " <- RARA (some sob ruído DP)" if p < 5 else ""
            print(f"    {cl:28s}: {p:5.1f}%{raro}")

        # Se houver hospitais, mostra heterogeneidade dessas classes
        if COL_HOSPITAL in df.columns and df[COL_HOSPITAL].nunique() > 1:
            tab_c = pd.crosstab(df[COL_HOSPITAL], df["_classe_atual"],
                                normalize="index") * 100
            het_atual = tab_c.std().mean()
            print(f"\n  Heterogeneidade das classes ATUAIS entre hospitais: "
                  f"{het_atual:.1f} pontos %")

        # Compara com os clusters descobertos, se disponíveis
        if labels_cluster is not None:
            print("\n  CLUSTERS descobertos (balanceamento):")
            dc = labels_cluster.value_counts(normalize=True).sort_index() * 100
            for cl, p in dc.items():
                print(f"    cluster {cl}: {p:5.1f}%")
            print("\n  >>> Compare: os clusters são mais equilibrados que as "
                  "classes atuais?")
            print("  >>> Se sim, e se separarem bem o desfecho (Parte 2), "
                  "são candidatos melhores.")
    else:
        print("  (colunas necessárias ausentes; pulei 4c)")

    # Encerra o "tee": restaura a saída normal e fecha o arquivo de análises
    sys.stdout = _stdout_original
    _arquivo_analises.close()
    print(f"\n  Análises da Parte 4 salvas em: {saida_analises}")

print("\n" + "=" * 70)
print(f"PIPELINE CONCLUÍDA — arquivos com carimbo {CARIMBO}")
print("=" * 70)
