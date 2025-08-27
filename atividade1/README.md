# MAX-SC-QBF: Problema de Maximização Quadrática com Restrição de Cobertura de Conjuntos

Este repositório contém a implementação de um **modelo de otimização combinatória** para resolver o problema **MAX-QBF (Maximização de Função Quadrática Binária)** com restrições adicionais de **Set Cover**, resultando na formulação conhecida como **MAX-SC-QBF**.  

A resolução é feita utilizando o **solver Gurobi**, com linearização da função objetivo quadrática para torná-la solucionável via programação inteira mista (MILP).

---

## 📌 Descrição do Problema

O **MAX-QBF** busca maximizar uma função quadrática binária da forma:

$$
Z = \sum_{i=1}^{n}\sum_{j=1}^{n} a_{ij} \cdot x_i \cdot x_j
$$

onde \(x_i \in \{0,1\}\).

A versão **MAX-SC-QBF** adiciona **restrições de cobertura de conjuntos (Set Cover)**, garantindo que todos os elementos do universo sejam cobertos por pelo menos um subconjunto escolhido.

---

## ⚙️ Estrutura do Código

O projeto contém as seguintes partes principais:

1. **Geração de instâncias (`gerar_instance`)**  
   - Cria instâncias aleatórias do problema, gerando subconjuntos que cobrem todos os elementos do universo.
   - Gera também os coeficientes da matriz quadrática \(A\), utilizada na função objetivo.

2. **Resolução do modelo (`solve_max_sc_qbf_linearized`)**  
   - Constrói um modelo no **Gurobi** com:
     - Variáveis binárias \(x_i\): seleção dos subconjuntos.
     - Variáveis binárias \(y_{ij}\): linearização do produto \(x_i \cdot x_j\).
   - Define:
     - **Função objetivo linearizada**.
     - **Restrições de linearização**.
     - **Restrições de cobertura de conjuntos**.
   - Executa a otimização com limite de tempo de **10 minutos**.

3. **Execução principal**  
   - Solicita ao usuário o valor de \(n\).
   - Gera e salva a instância em arquivo.
   - Chama o solver e registra os resultados em arquivo de log.

---

## 📊 Exemplo de Uso

```bash
Digite o valor de n: 5
--- Executando com n fornecido acima ---
```

Isso irá:
- Criar uma instância aleatória com 5 variáveis.
- Salvar a instância em `instancia5variaveis.txt`.
- Rodar a otimização e salvar os resultados em `log5variaveis.txt`.

---

## 📦 Dependências

- [Python 3.8+](https://www.python.org/)
- [Gurobi Optimizer](https://www.gurobi.com/)  
- Biblioteca Python do Gurobi:

```bash
pip install gurobipy
```

---

## 📑 Saída do Modelo

Dependendo do status da otimização, a saída pode apresentar:

- **Solução ótima encontrada**: valor de \(Z\) e subconjuntos selecionados.  
- **Solução dentro do limite de tempo**: melhor solução encontrada até o momento e *gap de otimalidade*.  
- **Modelo inviável**: quando não existe solução viável.  

---

## 🚀 Objetivo Educacional

Este código foi desenvolvido como parte de uma atividade acadêmica sobre **Otimização Combinatória** e **Programação Inteira Mista**, explorando:
- Linearização de funções objetivo quadráticas.
- Integração entre **MAX-QBF** e **Set Cover**.
- Uso do **Gurobi** para resolver problemas complexos de otimização.

---
