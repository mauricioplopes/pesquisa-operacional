# GRASP MAX-SC-QBF

Implementação do algoritmo GRASP (Greedy Randomized Adaptive Search Procedure) para o problema MAX-SC-QBF (Maximization of Quadratic Binary Function with Set Cover constraints).

## Estrutura do Projeto

```text
grasp_maxsc_qbf/
├── init.py              # Pacote principal
├── core/                    # Componentes base
│   ├── init.py
│   ├── solution.py         # Classe Solution genérica
│   ├── evaluator.py        # Evaluator
│   └── abstract_grasp.py   # GRASP abstrato
├── problems/                # Implementações de problemas
│   ├── init.py
│   └── qbf_sc.py          # Problema QBF-SC
├── algorithms/              # Algoritmos específicos
│   ├── init.py
│   └── grasp_qbf_sc.py    # GRASP para QBF-SC
├── utils/                   # Utilitários
│   ├── init.py
│   └── instance_generator.py # Gera instâncias para experimentos
└── main.py                  # Script principal
```

## Uso
### Execução básica
```bash
python main.py instance.txt
```

### Com parâmetros personalizados
```bash
python main.py (path)\instance.txt 0.3 100 standard first_improving
```

### Executar experimentos
```bash
python main.py --experiment
```

### Formato da Instância
```text
<n>                           # número de variáveis binárias
<s1> <s2> ... <sn>           # tamanhos dos subconjuntos
<elementos de S1>            # elementos cobertos por S1
<elementos de S2>            # elementos cobertos por S2
...
<elementos de Sn>            # elementos cobertos por Sn
<a11> <a12> ... <a1n>       # matriz QBF (triangular superior)
<a22> ... <a2n>
...
<ann>
```

### Configurações Disponíveis
- Métodos de Construção: standard, random_plus_greedy, sampled_greedy
- Busca Local: first_improving, best_improving
- Parâmetros: alpha (0.0-1.0), iterations (número de iterações)

