"""
=============================================================================
PIPELINE DE DESCOBERTA DE CLASSES v6 — MOSAIC-FL
ROBUSTEZ À PRIVACIDADE DIFERENCIAL (o critério que de fato importa)
=============================================================================

PROBLEMA REAL (medido no modelo de produção):
    As 5 classes atuais dão ~70% de acurácia SEM privacidade, mas caem para
    ~44% quando o ruído de DP é aplicado (federado, FedProx/FedNova + ruído).
    Uma queda de 26 pontos torna o modelo privado inutilizável.
    CAUSA: classes RARAS (melhora_pronto ~0,2%) colapsam sob ruído — têm pouco
    sinal, e o ruído de DP o engole. Classes próximas entre si também se tornam
    indistinguíveis quando o ruído chega.

OBJETIVO DO v6:
    Encontrar esquemas de classes cuja acurácia CAIA POUCO sob DP — mesmo que a
    base não seja a mais alta, o que importa é a acurácia COM privacidade.
    O v6 estende o placar do v5 com uma métrica DIRETA de robustez a DP,
    SEM treinar modelo pesado.

COMO MEDIMOS ROBUSTEZ SEM TREINAR (proxy, Parte 6):
    1. Para cada esquema, calcula os CENTROS de cada classe nas features
       numéricas padronizadas.
    2. Acurácia proxy = balanced_accuracy classificando cada ponto pelo centro
       de classe mais próximo. Usamos BALANCED accuracy (média do acerto por
       classe) porque ela NÃO é enganada pela classe majoritária — uma classe
       rara que some derruba a métrica, capturando o efeito real do DP.
    3. Injeta ruído gaussiano calibrado por epsilon (escala ~ 1/eps), AMPLIFICADO
       para classes raras (fator 1/sqrt(tamanho)) — reproduzindo o colapso de
       classes pequenas sob DP.
    4. Reporta acurácia SEM ruído e COM ruído (média de várias realizações).
    O melhor esquema é o de maior ACURÁCIA COM DP (absoluta).

    Mostra DUAS variantes do ruído (a pedido):
      - TEÓRICO PURO: escala derivada só do epsilon.
      - CALIBRADO: escala ajustada para o esquema E0 (classes atuais) reproduzir
        aproximadamente a queda real conhecida (70% -> 44%), tornando os demais
        esquemas comparáveis a esse ponto de referência.

    RESSALVA (honestidade metodológica): isto é um PROXY COMPARATIVO para
    ranquear esquemas rapidamente, não uma previsão do número exato. A validação
    final é treinar o esquema vencedor no pipeline federado real.

ESTRUTURA:
    Partes 0-4 — herdadas do v5
    Parte 5 — placar de esquemas (balanceamento, hospital, pureza)
    Parte 6 — ROBUSTEZ A DP (acurácia sem/com ruído por esquema)  <-- novidade

Saídas versionadas (VERSÃO + data/hora), incluindo:
    robustez_dp_V6_...txt   e   robustez_dp_V6_...csv

Dependências:
    pip install pandas numpy scikit-learn sqlalchemy psycopg2-binary fg-data-profiling kmodes
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
RODAR_PLACAR      = True    # Parte 5 (comparador de esquemas)
RODAR_ROBUSTEZ_DP = True    # Parte 6 (robustez à privacidade diferencial)

# --- Privacidade diferencial (Parte 6) ---
EPSILON = 1.0               # seu epsilon: menor = mais privacidade/ruído (ajuste!)
DP_N_REALIZACOES = 30       # média de N sorteios de ruído (estabilidade da métrica)
# Calibração: acurácia real conhecida das 5 classes atuais (E0), sem e com DP.
# Usada para ancorar a variante CALIBRADA do proxy ao seu ponto de referência.
DP_E0_BASE_REAL = 0.70      # 70% sem privacidade
DP_E0_DP_REAL   = 0.44      # 44% com privacidade

# Fonte: CSV LONGO (1 linha/exame) com attendance_id, ou leitura do banco.
# IMPORTANTE: para agregar corretamente, precisamos de attendance_id e de TODOS
# os exames de cada atendimento sorteado (ver estratégia de duas etapas abaixo).
CSV_ENTRADA = None
CONEXAO = "postgresql://mosaicfl:senhaForte@localhost:5432/mosaicfl"

# Amostragem UNIFORME (v4) + ESTRATIFICADA POR HOSPITAL (v5):
# sorteamos a MESMA quantidade de atendimentos de CADA hospital, limitada pelo
# hospital com MENOS dados. Isso torna o critério de equilíbrio entre hospitais
# (Parte 5) mensurável e justo — sem um hospital dominar a amostra.
LIMITE_ATENDIMENTOS = 40000    # teto TOTAL de atendimentos (dividido entre hospitais)
ESTRATIFICAR_HOSPITAL = True   # True = mesma qtd por hospital (recomendado p/ o placar)
LIMITE_EXPLORACAO   = 75000    # atendimentos p/ o relatório (guloso de memória)

# Clustering
COLUNA_ALVO = "outcome_class"
K_MIN, K_MAX = 2, 6            # k=7-8 não agregam (cotovelo ~6 nos dados reais)
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
VERSAO = "V6"    # identifica qual pipeline gerou o arquivo (ex: robustez_dp_V6_2026-07-06_1145.txt)
CARIMBO = datetime.now().strftime("%Y-%m-%d_%H%M")
print(f"Execução {VERSAO} (robustez à privacidade diferencial): {CARIMBO}\n")

def nome_saida(base, ext):
    # Formato: <base>_<VERSAO>_<data-hora>.<ext>  ->  relatorio_agg_V4_2026-07-06_1145.html
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
    """
    Carga em DUAS ETAPAS para amostragem uniforme por atendimento:
      1. Sorteia LIMITE_ATENDIMENTOS ids distintos (cada um com igual chance).
      2. Puxa TODOS os exames desses atendimentos (resumos completos).
    A subquery faz as duas coisas no banco, sem trafegar milhares de ids de volta.
    """
    if CSV_ENTRADA and os.path.exists(CSV_ENTRADA):
        print(f"Lendo formato longo do CSV: {CSV_ENTRADA}")
        return pd.read_csv(CSV_ENTRADA)

    print(f"Amostrando até {LIMITE_ATENDIMENTOS:,} atendimentos distintos "
          "e puxando TODOS os seus exames...")
    if ESTRATIFICAR_HOSPITAL:
        print("  (estratificado: mesma quantidade por hospital, limitado pelo menor)")
    else:
        print("  (amostragem uniforme por atendimento — resumos completos)")
    engine = create_engine(CONEXAO)

    if ESTRATIFICAR_HOSPITAL:
        # Estratificação por hospital: ROW_NUMBER() reinicia a contagem em cada
        # hospital (PARTITION BY), ordenada aleatoriamente. Pegando as primeiras
        # 'por_hosp' linhas de CADA hospital, todos contribuem igualmente.
        # 'por_hosp' é calculado DENTRO do SQL: divide o teto total pelo número
        # REAL de hospitais (COUNT DISTINCT) e limita pelo menor hospital.
        # Assim funciona para 2, 4 ou qualquer número de hospitais, sem constante.
        cte_amostra = f"""
        WITH elegiveis AS (
            SELECT a.attendance_id, a.hospital_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY a.hospital_id ORDER BY random()
                   ) AS rn,
                   COUNT(*)   OVER (PARTITION BY a.hospital_id) AS n_hosp
            FROM clinical.attendances          a
            JOIN metrics.clinical_outcomes     co ON co.attendance_id = a.attendance_id
            WHERE (co.outcome_at - a.attended_at) >= 0
              AND EXISTS (
                    SELECT 1 FROM metrics.exam_records e
                    WHERE e.attendance_id = a.attendance_id
                      AND e.analyte IS NOT NULL
                      AND e.classification IS NOT NULL
              )
        ),
        limite AS (
            -- por_hosp = min(teto/nº hospitais, tamanho do menor hospital).
            -- O nº de hospitais é contado em tempo real, então 2 ou 4 dá certo.
            SELECT LEAST(
                     {LIMITE_ATENDIMENTOS} / COUNT(DISTINCT hospital_id),
                     MIN(n_hosp)
                   ) AS por_hosp
            FROM elegiveis
        ),
        amostra AS (
            SELECT e.attendance_id
            FROM elegiveis e, limite l
            WHERE e.rn <= l.por_hosp
        )
        """
    else:
        cte_amostra = f"""
        WITH amostra AS (
            SELECT a.attendance_id
            FROM clinical.attendances          a
            JOIN metrics.clinical_outcomes     co ON co.attendance_id = a.attendance_id
            WHERE (co.outcome_at - a.attended_at) >= 0
              AND EXISTS (
                    SELECT 1 FROM metrics.exam_records e
                    WHERE e.attendance_id = a.attendance_id
                      AND e.analyte IS NOT NULL
                      AND e.classification IS NOT NULL
              )
            ORDER BY random()
            LIMIT {LIMITE_ATENDIMENTOS}
        )
        """

    query = cte_amostra + """
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
        FROM amostra                       s
        JOIN clinical.attendances          a  ON a.attendance_id  = s.attendance_id
        JOIN clinical.patients             p  ON p.patient_id     = a.patient_id
        JOIN metrics.clinical_outcomes     co ON co.attendance_id = a.attendance_id
        JOIN metrics.exam_records          e  ON e.attendance_id  = a.attendance_id
        WHERE e.analyte IS NOT NULL
          AND e.classification IS NOT NULL
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
    print("PARTE 0 — VIÉS DE GRANULARIDADE e a correção do v4")
    print("=" * 70)

    # Exames por atendimento: a fonte do viés
    ex_por_atend = long_df.groupby(COL_ATENDIMENTO).size()
    print("\nExames por atendimento NESTA AMOSTRA:")
    print(f"  mínimo: {ex_por_atend.min()} | mediana: {ex_por_atend.median():.0f} "
          f"| média: {ex_por_atend.mean():.1f} | máximo: {ex_por_atend.max()}")

    # Ganho central do v4: amostra uniforme + resumos completos
    print("\nComo o v4 corrige o viés residual do v3:")
    print("  v3: 'LIMIT' nas linhas de exame -> atendimentos com muitos exames")
    print("      têm mais chance de entrar, e resumos de atendimentos pequenos")
    print("      ficam incompletos (poucas linhas capturadas).")
    print("  v4: sorteia ATENDIMENTOS distintos (chance igual p/ todos) e traz")
    print("      TODOS os exames de cada um -> resumos 100% completos.")
    print(f"\n  Confirmação nesta amostra: {len(df):,} atendimentos, cada um com")
    print("  todos os seus exames (nenhum resumo parcial). A chance de entrar na")
    print("  amostra NÃO depende mais do número de exames do atendimento.")

    # Distribuição do desfecho: por linha de exame vs por atendimento.
    # (Aqui as diferenças refletem quão desiguais são os nº de exames por
    #  desfecho — se um desfecho tende a gerar mais exames, ele domina o longo.)
    print("\nDistribuição de 'outcome_class': por EXAME vs por ATENDIMENTO:")
    dist_long = long_df[COLUNA_ALVO].value_counts(normalize=True).sort_index() * 100
    dist_agg = df[COLUNA_ALVO].value_counts(normalize=True).sort_index() * 100
    comp = pd.DataFrame({"por_exame_%": dist_long.round(1),
                         "por_atendimento_%": dist_agg.round(1)})
    comp["diferença"] = (comp["por_atendimento_%"] - comp["por_exame_%"]).round(1)
    print("  " + comp.to_string().replace("\n", "\n  "))
    print("\n  >>> A coluna 'por_atendimento_%' é a distribuição REAL e sem viés.")
    print("  >>> Diferenças grandes indicam desfechos que geram mais/menos exames.")

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

    # Detecção ROBUSTA de coluna categórica, compatível com qualquer versão do
    # pandas. No pandas 3.0 strings viram dtype 'str' (não 'object'), então a
    # checagem antiga (dtype == object) falhava e o k-prototypes tratava texto
    # como número — provável causa do erro de inicialização. Aqui consideramos
    # categórica tudo que NÃO for numérico.
    def eh_categorica(serie):
        return not pd.api.types.is_numeric_dtype(serie)

    alvo = df[COLUNA_ALVO].copy() if COLUNA_ALVO in df.columns else None

    # Colunas fora do clustering: IDs, datas, e o alvo
    descartar = {COL_ATENDIMENTO, COL_PACIENTE, COL_ATENDIDO, COL_DESFECHO, COLUNA_ALVO}
    X = df.drop(columns=[c for c in descartar if c in df.columns], errors="ignore")

    # Trata faltantes; age==0 -> ausente (idade disfarçada)
    for col in X.columns:
        if col == "age":
            X[col] = pd.to_numeric(X[col], errors="coerce").replace(0, np.nan)
            X[col] = X[col].fillna(X[col].median())
        elif eh_categorica(X[col]):
            X[col] = X[col].astype(str).fillna("DESCONHECIDO")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())

    # Remove categóricas de cardinalidade MUITO ALTA. Elas travam a
    # inicialização do k-prototypes (não há como formar protótipos distintos
    # quando quase todo valor é único) e não ajudam a separar grupos.
    MAX_CARD_CAT = 50
    cat_altas = []
    for col in list(X.columns):
        if eh_categorica(X[col]) and X[col].nunique() > MAX_CARD_CAT:
            cat_altas.append((col, X[col].nunique()))
            X = X.drop(columns=[col])
    if cat_altas:
        print("  Removidas categóricas de alta cardinalidade (travam o cluster):")
        for c, n in cat_altas:
            print(f"    - {c} ({n} categorias)")

    idx_cat = [i for i, c in enumerate(X.columns) if eh_categorica(X[c])]
    print(f"  {X.shape[1]} colunas ({len(idx_cat)} categóricas, "
          f"{X.shape[1]-len(idx_cat)} numéricas)")
    print(f"  Features de exame agregadas: n_exames, n_high, n_low, n_normal, "
          f"n_analitos_dist, prop_alterados")
    print(f"  Colunas: {list(X.columns)}\n")

    # Matriz com tipos limpos: numéricas como float, categóricas como str.
    # Tipos mistos/objetos ambíguos são outra causa de falha de inicialização.
    for c in X.columns:
        if eh_categorica(X[c]):
            X[c] = X[c].astype(str)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    matriz = X.to_numpy()
    print(f"  Testando k de {K_MIN} a {K_MAX}:")
    print(f"  {'k':>3} | {'custo':>15}")
    print("  " + "-" * 23)

    def clusterizar(k):
        """Tenta 'Cao' (estável); se falhar, cai para 'Huang' e depois 'random'.
        Retorna None se nenhuma inicialização funcionar (k alto demais para os
        dados — não há como formar k protótipos distintos)."""
        for init in ("Cao", "Huang", "random"):
            try:
                kp = KPrototypes(n_clusters=k, init=init, random_state=42,
                                 n_init=(1 if init == "random" else 2), verbose=0)
                lab = kp.fit_predict(matriz, categorical=idx_cat)
                return kp, lab, init
            except ValueError:
                continue
        return None  # nenhuma init funcionou para este k

    resultados = {}
    for k in range(K_MIN, K_MAX + 1):
        r = clusterizar(k)
        if r is None:
            print(f"  {k:>3} | (não convergiu — k alto demais; ignorado)")
            continue
        kp, lab, init_usado = r
        resultados[k] = {"labels": lab, "custo": kp.cost_}
        extra = "" if init_usado == "Cao" else f"  (init={init_usado})"
        print(f"  {k:>3} | {kp.cost_:>15,.0f}{extra}")

    if not resultados:
        print("\n  ! Nenhum k convergiu. Verifique as features (ver auditoria).")
        raise SystemExit(1)

    k_usar = K_ESCOLHIDO if K_ESCOLHIDO in resultados else max(resultados)
    if k_usar != K_ESCOLHIDO:
        print(f"\n  Aviso: k={K_ESCOLHIDO} não convergiu; usando k={k_usar}.")
    melhor = resultados[k_usar]
    K_ESCOLHIDO = k_usar
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


# =============================================================================
# PARTE 5 — PLACAR COMPARATIVO DE ESQUEMAS DE CLASSIFICAÇÃO
# =============================================================================
# Dict compartilhado: a Parte 5 preenche, a Parte 6 (robustez DP) reutiliza.
esquemas_compartilhados = {}

if RODAR_PLACAR:
    saida_placar = nome_saida("placar_esquemas", "txt")
    arq = open(saida_placar, "w")
    orig = sys.stdout
    sys.stdout = _Tee(orig, arq)

    print("=" * 70)
    print("PARTE 5 — PLACAR COMPARATIVO DE ESQUEMAS DE CLASSES")
    print(f"Execução: {CARIMBO}")
    print("=" * 70)

    # --- As 3 métricas do placar ---
    def m_balanceamento(classes):
        """Tamanho da MENOR classe (%). Maior = melhor (evita classes raras)."""
        d = pd.Series(classes).value_counts(normalize=True) * 100
        return d.min()

    def m_equilibrio_hosp(classes, hosp):
        """Variação média das classes entre hospitais. MENOR = melhor (menos non-IID)."""
        if hosp is None or hosp.nunique() < 2:
            return np.nan
        t = pd.crosstab(hosp, classes, normalize="index") * 100
        return t.std().mean()

    def m_preditiva(classes, outcome):
        """Pureza média das classes quanto ao desfecho. Maior = melhor."""
        t = pd.crosstab(classes, outcome, normalize="index") * 100
        return t.max(axis=1).mean()

    def n_classes(classes):
        return pd.Series(classes).nunique()

    hosp = df[COL_HOSPITAL] if COL_HOSPITAL in df.columns else None
    outcome = df[COLUNA_ALVO] if COLUNA_ALVO in df.columns else None
    tem_hosp = hosp is not None and hosp.nunique() >= 2

    if not tem_hosp:
        print("\n  ! AVISO: amostra tem só um hospital — critério de equilíbrio")
        print("  ! entre hospitais ficará indisponível (NaN). Verifique a query.")

    # -------------------------------------------------------------------------
    # Constrói os esquemas candidatos
    # (guardados em 'esquemas_compartilhados' para a Parte 6 reutilizar)
    # -------------------------------------------------------------------------
    esquemas = esquemas_compartilhados

    # Regra base das classes atuais, parametrizada pelo corte de duração
    def regra_classes(corte):
        def classificar(row):
            oc = row.get(COLUNA_ALVO)
            internado = "nternado" in str(row.get(COL_TIPO_ATEND, ""))
            dur = row.get("duration_days", np.nan)
            if oc == 0 and not internado: return "curado_pronto"
            if oc == 0 and internado:     return "curado_internado"
            if oc == 1 and not internado: return "melhora_pronto"
            if oc == 1 and internado:
                return "internado_breve" if dur <= corte else "internado_grave"
            return "outro"
        return df.apply(classificar, axis=1)

    # E0 — 5 classes atuais (corte 10, baseline)
    esquemas["E0_atual_10d"] = regra_classes(10)

    # E1 — cortes de duração alternativos
    for corte in [5, 7, 14, 21]:
        esquemas[f"E1_corte_{corte}d"] = regra_classes(corte)

    # E2 — clustering puro (usa os labels já calculados na Parte 2, se houver)
    if labels_cluster is not None:
        # Alinha labels do cluster ao índice de df (podem ter sido amostrados)
        cl = pd.Series("cluster_?", index=df.index)
        cl.loc[labels_cluster.index] = "cluster_" + labels_cluster.astype(str)
        esquemas[f"E2_cluster_k{K_ESCOLHIDO}"] = cl

    # E3 — incorpora os 'outro' (outcome 2-6) como classes próprias em vez de descartar
    def regra_com_outros(row):
        oc = row.get(COLUNA_ALVO)
        internado = "nternado" in str(row.get(COL_TIPO_ATEND, ""))
        dur = row.get("duration_days", np.nan)
        if oc == 0 and not internado: return "curado_pronto"
        if oc == 0 and internado:     return "curado_internado"
        if oc == 1 and not internado: return "melhora_pronto"
        if oc == 1 and internado:
            return "internado_breve" if dur <= 10 else "internado_grave"
        return f"outcome_{oc}"   # 2-6 viram classes próprias, não 'outro'
    esquemas["E3_com_outros"] = df.apply(regra_com_outros, axis=1)

    # -------------------------------------------------------------------------
    # Monta o placar
    # -------------------------------------------------------------------------
    print("\nEsquemas avaliados:")
    linhas_placar = []
    for nome, classes in esquemas.items():
        linhas_placar.append({
            "esquema": nome,
            "n_classes": n_classes(classes),
            "menor_classe_%": round(m_balanceamento(classes), 2),
            "heterog_hosp": (round(m_equilibrio_hosp(classes, hosp), 2)
                             if tem_hosp else np.nan),
            "pureza_desfecho_%": round(m_preditiva(classes, outcome), 1),
        })
    placar = pd.DataFrame(linhas_placar)

    # Rankings (1 = melhor em cada critério)
    placar["rank_balanc"] = placar["menor_classe_%"].rank(ascending=False).astype(int)
    if tem_hosp:
        placar["rank_hosp"] = placar["heterog_hosp"].rank(ascending=True).astype(int)
    placar["rank_predit"] = placar["pureza_desfecho_%"].rank(ascending=False).astype(int)

    # Score combinado (soma dos ranks; menor = melhor no conjunto)
    rank_cols = ["rank_balanc", "rank_predit"] + (["rank_hosp"] if tem_hosp else [])
    placar["score_total"] = placar[rank_cols].sum(axis=1)
    placar = placar.sort_values("score_total")

    print("\n" + "=" * 70)
    print("PLACAR (ordenado por score total; menor score = melhor no conjunto)")
    print("=" * 70)
    print("\nCritérios:")
    print("  menor_classe_%    : tamanho da menor classe (MAIOR = melhor)")
    print("  heterog_hosp      : variação entre hospitais (MENOR = melhor)")
    print("  pureza_desfecho_% : alinhamento com outcome (MAIOR = melhor)")
    print()
    print(placar.to_string(index=False))

    # Destaques
    print("\n--- LEITURA ---")
    melhor_geral = placar.iloc[0]
    print(f"Melhor no conjunto: {melhor_geral['esquema']} "
          f"(score {melhor_geral['score_total']})")
    base = placar[placar["esquema"] == "E0_atual_10d"].iloc[0]
    print(f"Baseline (5 classes atuais): menor classe {base['menor_classe_%']}%, "
          f"pureza {base['pureza_desfecho_%']}%")
    if melhor_geral["esquema"] != "E0_atual_10d":
        print(f">>> '{melhor_geral['esquema']}' supera as classes atuais no placar.")
        print(f">>> menor classe: {base['menor_classe_%']}% -> "
              f"{melhor_geral['menor_classe_%']}% "
              "(classes menos raras = mais robustas sob DP)")
    else:
        print(">>> As classes atuais lideram o placar nesta amostra.")

    # Salva o placar também como CSV para análise posterior
    placar.to_csv(nome_saida("placar_esquemas", "csv"), index=False)

    sys.stdout = orig
    arq.close()
    print(f"\n  Placar salvo em: {saida_placar}")
    print(f"  Placar (CSV) em: {nome_saida('placar_esquemas', 'csv')}")


# =============================================================================
# PARTE 6 — ROBUSTEZ À PRIVACIDADE DIFERENCIAL
# =============================================================================
if RODAR_ROBUSTEZ_DP:
    from sklearn.metrics import balanced_accuracy_score

    saida_dp = nome_saida("robustez_dp", "txt")
    arq = open(saida_dp, "w")
    orig = sys.stdout
    sys.stdout = _Tee(orig, arq)

    print("=" * 70)
    print("PARTE 6 — ROBUSTEZ À PRIVACIDADE DIFERENCIAL")
    print(f"Execução: {CARIMBO} | epsilon = {EPSILON}")
    print("=" * 70)

    # Se a Parte 5 não rodou, os esquemas não existem — avisa e pula.
    if not esquemas_compartilhados:
        print("\n  ! Parte 5 (placar) não rodou, então não há esquemas para testar.")
        print("  ! Ative RODAR_PLACAR = True para gerar os esquemas candidatos.")
        sys.stdout = orig
        arq.close()
    else:
        # Features NUMÉRICAS para calcular centros de classe (padronizadas)
        num_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])
                    and c not in (COLUNA_ALVO, "duration_days")]
        # inclui duration_days como feature preditiva (é informativa)
        if "duration_days" in df.columns:
            num_cols.append("duration_days")
        Xnum = df[num_cols].copy()
        for c in num_cols:
            Xnum[c] = pd.to_numeric(Xnum[c], errors="coerce")
            Xnum[c] = Xnum[c].fillna(Xnum[c].median())
        # padroniza (z-score) para o ruído ter escala comparável entre features
        Xz = ((Xnum - Xnum.mean()) / Xnum.std().replace(0, 1)).to_numpy()
        print(f"\n  Features numéricas usadas: {num_cols}")

        def acuracia_proxy(y, escala_ruido, rng, n_real):
            """
            balanced_accuracy classificando cada ponto pelo centro de classe
            mais próximo. Com escala_ruido>0, injeta ruído nos centros
            (amplificado para classes raras). Retorna média de n_real sorteios.
            """
            y = np.asarray(y)
            classes = pd.Series(y).unique()
            centros0 = np.vstack([Xz[y == c].mean(axis=0) for c in classes])
            if escala_ruido <= 0:
                dist = np.linalg.norm(Xz[:, None, :] - centros0[None, :, :], axis=2)
                pred = classes[dist.argmin(axis=1)]
                return balanced_accuracy_score(y, pred)
            tam = np.array([(y == c).mean() for c in classes])
            fator = 1.0 / np.sqrt(np.clip(tam, 1e-6, None))  # rara sofre mais
            accs = []
            for _ in range(n_real):
                ruido = rng.normal(0, escala_ruido, centros0.shape) * fator[:, None]
                centros = centros0 + ruido
                dist = np.linalg.norm(Xz[:, None, :] - centros[None, :, :], axis=2)
                pred = classes[dist.argmin(axis=1)]
                accs.append(balanced_accuracy_score(y, pred))
            return float(np.mean(accs))

        rng = np.random.default_rng(42)
        disp = Xz.std(axis=0).mean()  # dispersão média das features padronizadas

        # --- Variante 1: ruído TEÓRICO PURO (escala = disp / epsilon) ---
        escala_teorica = disp / max(EPSILON, 1e-6)

        # --- Variante 2: ruído CALIBRADO para E0 reproduzir 70%->44% ---
        # busca a escala que faz a acurácia_dp de E0 bater DP_E0_DP_REAL,
        # relativa à sua base proxy (regra de três com a base real conhecida).
        y_e0 = esquemas_compartilhados.get("E0_atual_10d")
        escala_calibrada = escala_teorica  # fallback
        if y_e0 is not None:
            base_proxy_e0 = acuracia_proxy(y_e0, 0, rng, 1)
            # alvo: reduzir a base_proxy na MESMA proporção do real (44/70)
            alvo_dp_e0 = base_proxy_e0 * (DP_E0_DP_REAL / DP_E0_BASE_REAL)
            # busca binária simples na escala de ruído
            lo, hi = 0.0, disp * 10
            for _ in range(25):
                mid = (lo + hi) / 2
                a = acuracia_proxy(y_e0, mid, rng, DP_N_REALIZACOES)
                if a > alvo_dp_e0:
                    lo = mid           # precisa de mais ruído
                else:
                    hi = mid
            escala_calibrada = (lo + hi) / 2
            print(f"\n  Calibração: E0 base_proxy={base_proxy_e0*100:.0f}% -> "
                  f"alvo_dp={alvo_dp_e0*100:.0f}% (espelha {DP_E0_BASE_REAL*100:.0f}"
                  f"->{DP_E0_DP_REAL*100:.0f} real)")
            print(f"  escala de ruído: teórica={escala_teorica:.3f}, "
                  f"calibrada={escala_calibrada:.3f}")

        # --- Monta a tabela de robustez ---
        linhas = []
        for nome, classes in esquemas_compartilhados.items():
            base = acuracia_proxy(classes, 0, rng, 1)
            dp_teo = acuracia_proxy(classes, escala_teorica, rng, DP_N_REALIZACOES)
            dp_cal = acuracia_proxy(classes, escala_calibrada, rng, DP_N_REALIZACOES)
            menor = (pd.Series(classes).value_counts(normalize=True) * 100).min()
            linhas.append({
                "esquema": nome,
                "n_classes": pd.Series(classes).nunique(),
                "menor_classe_%": round(menor, 2),
                "acc_base_%": round(base * 100, 1),
                "acc_dp_teorico_%": round(dp_teo * 100, 1),
                "acc_dp_calibrado_%": round(dp_cal * 100, 1),
                "queda_calibrada_pts": round((base - dp_cal) * 100, 1),
            })
        tab = pd.DataFrame(linhas)
        # ordena pelo que importa: maior acurácia COM DP (calibrada)
        tab = tab.sort_values("acc_dp_calibrado_%", ascending=False)

        print("\n" + "=" * 70)
        print("ROBUSTEZ A DP (ordenado por acurácia COM privacidade, calibrada)")
        print("=" * 70)
        print("\nProxy comparativo (balanced accuracy por centros de classe).")
        print("O que importa é 'acc_dp_calibrado_%': acurácia estimada COM privacidade.")
        print("Mostradas 2 variantes de ruído: teórica (só epsilon) e calibrada")
        print(f"(ancorada em E0 = {DP_E0_BASE_REAL*100:.0f}->{DP_E0_DP_REAL*100:.0f}).\n")
        print(tab.to_string(index=False))

        # Leitura
        print("\n--- LEITURA ---")
        melhor = tab.iloc[0]
        e0row = tab[tab["esquema"] == "E0_atual_10d"]
        print(f"Esquema mais robusto sob DP: {melhor['esquema']}")
        print(f"  acurácia com privacidade (calibrada): {melhor['acc_dp_calibrado_%']}%")
        if len(e0row):
            e0 = e0row.iloc[0]
            print(f"Baseline E0 (classes atuais): {e0['acc_dp_calibrado_%']}% com DP")
            if melhor["esquema"] != "E0_atual_10d":
                ganho = melhor["acc_dp_calibrado_%"] - e0["acc_dp_calibrado_%"]
                print(f">>> '{melhor['esquema']}' mantém +{ganho:.1f} pontos sob DP")
                print(">>> que as classes atuais — candidato mais robusto à privacidade.")
            else:
                print(">>> As classes atuais já são as mais robustas nesta amostra.")

        print("\n(!) Proxy comparativo p/ ranquear esquemas — não é o número final.")
        print("    Valide o esquema vencedor treinando no seu pipeline federado real.")

        tab.to_csv(nome_saida("robustez_dp", "csv"), index=False)
        sys.stdout = orig
        arq.close()
        print(f"\n  Robustez DP salva em: {saida_dp}")
        print(f"  Robustez DP (CSV) em: {nome_saida('robustez_dp', 'csv')}")


print("\n" + "=" * 70)
print(f"PIPELINE {VERSAO} CONCLUÍDA — arquivos com carimbo {CARIMBO}")
print("=" * 70)
