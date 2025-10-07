from pathlib import Path

# ========================= Caminhos base =========================
# Define a BASE dinâmica como a pasta do repositório (onde este arquivo reside)
BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "data/output"

# ========================= Arquivos de entrada/saída =========================
# CSV bruto de entrada
INPUT_SITAF = BASE / "data/input/sitaf-dados-dt-final.csv"
INPUT_ESTATISTICAS = BASE / "data/input/estatisticas-processos.csv"

# Prefixo de saída do pré-processamento (gera .csv, .npz, .meta.json)
OUTPUT_SIFAF = OUT_DIR / "preprocessado"

# Artefato .npz produzido no pré-processamento
INPUT_PREPROCESSADO = OUTPUT_SIFAF.with_suffix(".npz")

# Prefixos de saída dos runners
OUTPUT_PREFIXO_DP = OUT_DIR / "dp"
OUTPUT_PREFIXO_GULOSO = OUT_DIR / "guloso"
OUTPUT_PREFIXO_AG = OUT_DIR / "ga"

# ========================= Parâmetros operacionais =========================
NUM_PROCURADORES = 30
HORAS_DIA = 8
DIAS_UTEIS = 220
CAPACIDADE_PADRAO = float(NUM_PROCURADORES * HORAS_DIA * DIAS_UTEIS)

# Resolução (em horas) para discretização na DP
RESOLUCAO_PADRAO = 0.25

# ========================= Heurística de esforço =========================
# Fatores multiplicativos por tipo de tributo (heurística de esforço)
TIPO_FATOR = {
    "ICMS": 1.2,
    "IPVA": 1.0,
    "ISS": 1.1,
    "ITCD": 1.3,
    "OUTROS": 1.0,
}

# ========================= Hiperparâmetros do AG =========================
POPULACAO = 80
GERACOES = 150
TAXA_CRUZAMENTO = 0.7
TAXA_MUTACAO = 0.02
SEMENTE = 42
