# Seleção de Processos de Dívida Ativa (PGDF)

Preparação de dados e seleção de processos formulada como Problema da Mochila, com comparação entre três métodos: Programação Dinâmica (DP), Guloso e Algoritmo Genético (AG).

## Visão geral
- **Objetivo**: maximizar a arrecadação respeitando a capacidade operacional (horas disponíveis).
- **Algoritmos**:
  - Programação Dinâmica (0-1, com discretização de horas)
  - Heurística Gulosa (razão valor/peso)
  - Algoritmo Genético (bitstring)

## Ambiente
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dados
- Entrada padrão: `data/input/sitaf-dados-dt-final.csv` (separador `;`, decimal `,`).
- Estatísticas auxiliares: `data/input/estatisticas-processos.csv`.
- Caminhos e parâmetros padrão ficam em `config.py`.

## Pré-processamento
Gera `preprocessado.csv`, `preprocessado.npz` e `preprocessado.meta.json` em `data/output/`.

```bash
python preprocess.py \
  --entrada data/input/sitaf-dados-dt-final.csv \
  --prefixo-saida data/output/preprocessado
```

Se omitido, usa os caminhos padrão de `config.py`.

## Execução dos algoritmos
Os três scripts consomem `preprocessado.npz` (default em `data/output/preprocessado.npz`). Parâmetros default também vêm de `config.py`.

### Programação Dinâmica (DP)
Gera `data/output/dp.csv` e `data/output/dp.json`.
```bash
python run_dp.py \
  --capacidade 52800 \
  --resolucao 0.25 \
  --top-k 0 \
  --modo-filtro ratio \
  --npz data/output/preprocessado.npz \
  --prefixo-saida data/output/dp
```

Parâmetros úteis:
- `--resolucao`: discretização em horas (ex.: 0.25)
- `--top-k`: opcional para reduzir universo (0 desliga)
- `--modo-filtro`: `none`, `value` ou `ratio`

### Guloso
Gera `data/output/guloso.csv` e `data/output/guloso.json`.
```bash
python run_greedy.py \
  --capacidade 52800 \
  --npz data/output/preprocessado.npz \
  --prefixo-saida data/output/guloso
```

### Algoritmo Genético (AG)
Gera `data/output/ga.csv` e `data/output/ga.json`.
```bash
python run_ga.py \
  --capacidade 52800 \
  --pop 80 \
  --gens 150 \
  --cxpb 0.7 \
  --mutpb 0.02 \
  --seed 42 \
  --npz data/output/preprocessado.npz \
  --prefixo-saida data/output/ga
```

## Saídas
- Pré-processamento: `data/output/preprocessado.{csv,npz,meta.json}`
- DP: `data/output/dp.{csv,json}`
- Guloso: `data/output/guloso.{csv,json}`
- AG: `data/output/ga.{csv,json}`

## Configuração
Edite `config.py` para ajustar:
- Caminhos de entrada/saída
- Capacidade padrão (nº procuradores, horas/dia, dias úteis)
- Resolução da DP
- Hiperparâmetros do AG

## Observações
- O peso operacional por processo é heurístico e calculado no pré-processamento (`peso_horas`).
- Todos os scripts imprimem no stdout um resumo JSON da execução.