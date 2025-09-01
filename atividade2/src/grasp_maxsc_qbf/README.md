# Pseudo-código dos Métodos GRASP MAX-SC-QBF
## 1. Elaboração do RCL (Restricted Candidate List)
### Método Standard RCL Construction
```text
ALGORITMO: Construir_RCL_Standard(CL, sol, alpha)
ENTRADA: CL (lista de candidatos), sol (solução atual), alpha (parâmetro)
SAÍDA: RCL (lista restrita de candidatos)

INÍCIO
    max_cost ← -∞
    min_cost ← +∞
    RCL ← ∅
    
    // Avaliar todos os candidatos para encontrar min e max
    PARA CADA c ∈ CL FAÇA
        delta_cost ← avaliar_custo_inserção(c, sol)
        SE delta_cost < min_cost ENTÃO
            min_cost ← delta_cost
        FIM SE
        SE delta_cost > max_cost ENTÃO
            max_cost ← delta_cost
        FIM SE
    FIM PARA
    
    // Construir RCL com candidatos dentro do threshold
    threshold ← max_cost - alpha × (max_cost - min_cost)
    
    PARA CADA c ∈ CL FAÇA
        delta_cost ← avaliar_custo_inserção(c, sol)
        SE delta_cost ≥ threshold ENTÃO  // Para maximização
            RCL ← RCL ∪ {c}
        FIM SE
    FIM PARA
    
    RETORNAR RCL
FIM
```

## 2. Métodos de Construção
### 2.1 Standard Construction
```text
ALGORITMO: Construção_Standard(alpha, iterações)
ENTRADA: alpha (parâmetro guloso), iterações (número máximo)
SAÍDA: sol (solução construída)

INÍCIO
    CL ← fazer_lista_candidatos()
    sol ← criar_solução_vazia()
    
    ENQUANTO NÃO critério_parada(sol) FAÇA
        atualizar_CL(sol)
        
        SE CL = ∅ ENTÃO
            INTERROMPER
        FIM SE
        
        RCL ← construir_RCL_standard(CL, sol, alpha)
        
        SE RCL ≠ ∅ ENTÃO
            índice_aleatório ← aleatório(0, |RCL|-1)
            candidato_escolhido ← RCL[índice_aleatório]
            
            CL ← CL \ {candidato_escolhido}
            sol ← sol ∪ {candidato_escolhido}
            
            avaliar(sol)
        FIM SE
    FIM ENQUANTO
    
    RETORNAR sol
FIM
```

### 2.2 Random Plus Greedy Construction
```text
ALGORITMO: Construção_Random_Plus_Greedy()
ENTRADA: Nenhuma específica
SAÍDA: sol (solução construída)

INÍCIO
    CL ← fazer_lista_candidatos()
    sol ← criar_solução_vazia()
    
    // FASE 1: Seleção Aleatória (30% dos elementos)
    num_aleatórios ← max(1, ⌊0.3 × |CL|⌋)
    
    PARA i ← 1 ATÉ min(num_aleatórios, |CL|) FAÇA
        SE CL = ∅ ENTÃO
            INTERROMPER
        FIM SE
        
        índice_aleatório ← aleatório(0, |CL|-1)
        selecionado ← CL[índice_aleatório]
        CL ← CL \ {selecionado}
        sol ← sol ∪ {selecionado}
    FIM PARA
    
    // FASE 2: Conclusão Gulosa
    ENQUANTO NÃO critério_parada(sol) E CL ≠ ∅ FAÇA
        atualizar_CL(sol)
        
        SE CL = ∅ ENTÃO
            INTERROMPER
        FIM SE
        
        melhor_candidato ← NULO
        melhor_custo ← -∞
        
        PARA CADA c ∈ CL FAÇA
            delta_custo ← avaliar_custo_inserção(c, sol)
            SE delta_custo > melhor_custo ENTÃO
                melhor_custo ← delta_custo
                melhor_candidato ← c
            FIM SE
        FIM PARA
        
        SE melhor_candidato ≠ NULO ENTÃO
            CL ← CL \ {melhor_candidato}
            sol ← sol ∪ {melhor_candidato}
            avaliar(sol)
        FIM SE
    FIM ENQUANTO
    
    RETORNAR sol
FIM
```

### 2.3 Sampled Greedy Construction
```text
ALGORITMO: Construção_Sampled_Greedy()
ENTRADA: Nenhuma específica
SAÍDA: sol (solução construída)

INÍCIO
    CL ← fazer_lista_candidatos()
    sol ← criar_solução_vazia()
    tamanho_amostra ← max(2, ⌊0.5 × |CL|⌋)  // 50% dos candidatos
    
    ENQUANTO NÃO critério_parada(sol) E CL ≠ ∅ FAÇA
        atualizar_CL(sol)
        
        SE CL = ∅ ENTÃO
            INTERROMPER
        FIM SE
        
        // Amostrar candidatos
        tamanho_atual ← min(tamanho_amostra, |CL|)
        candidatos_amostrados ← amostra_aleatória(CL, tamanho_atual)
        
        // Encontrar melhor entre os amostrados
        melhor_candidato ← NULO
        melhor_custo ← -∞
        
        PARA CADA c ∈ candidatos_amostrados FAÇA
            delta_custo ← avaliar_custo_inserção(c, sol)
            SE delta_custo > melhor_custo ENTÃO
                melhor_custo ← delta_custo
                melhor_candidato ← c
            FIM SE
        FIM PARA
        
        SE melhor_candidato ≠ NULO ENTÃO
            CL ← CL \ {melhor_candidato}
            sol ← sol ∪ {melhor_candidato}
            avaliar(sol)
        FIM SE
    FIM ENQUANTO
    
    RETORNAR sol
FIM
```

## 3. Métodos de Busca Local
### 3.1 First Improving Local Search
```text
ALGORITMO: Busca_Local_First_Improving(sol)
ENTRADA: sol (solução inicial)
SAÍDA: sol (solução melhorada)

INÍCIO
    melhorou ← VERDADEIRO
    
    ENQUANTO melhorou FAÇA
        melhorou ← FALSO
        melhor_movimento ← NULO
        
        CL_atual ← {i : i ∉ sol}  // Candidatos não na solução
        
        // AVALIAR INSERÇÕES
        PARA CADA cand_in ∈ CL_atual FAÇA
            delta_custo ← avaliar_custo_inserção(cand_in, sol)
            
            SE delta_custo > 0 ENTÃO  // Movimento melhora
                melhor_movimento ← ("inserir", cand_in, NULO)
                INTERROMPER  // First improving
            FIM SE
        FIM PARA
        
        SE melhor_movimento = NULO ENTÃO
            // AVALIAR REMOÇÕES
            PARA CADA cand_out ∈ sol FAÇA
                delta_custo ← avaliar_custo_remoção(cand_out, sol)
                
                SE delta_custo > 0 ENTÃO  // Movimento melhora
                    melhor_movimento ← ("remover", NULO, cand_out)
                    INTERROMPER  // First improving
                FIM SE
            FIM PARA
        FIM SE
        
        SE melhor_movimento = NULO ENTÃO
            // AVALIAR TROCAS
            PARA CADA cand_in ∈ CL_atual FAÇA
                PARA CADA cand_out ∈ sol FAÇA
                    delta_custo ← avaliar_custo_troca(cand_in, cand_out, sol)
                    
                    SE delta_custo > 0 ENTÃO  // Movimento melhora
                        melhor_movimento ← ("trocar", cand_in, cand_out)
                        INTERROMPER  // First improving
                    FIM SE
                FIM PARA
                
                SE melhor_movimento ≠ NULO ENTÃO
                    INTERROMPER
                FIM SE
            FIM PARA
        FIM SE
        
        // APLICAR MELHOR MOVIMENTO
        SE melhor_movimento ≠ NULO ENTÃO
            (tipo, cand_in, cand_out) ← melhor_movimento
            
            CASO tipo DE
                "inserir": sol ← sol ∪ {cand_in}
                "remover": sol ← sol \ {cand_out}
                "trocar": 
                    sol ← sol \ {cand_out}
                    sol ← sol ∪ {cand_in}
            FIM CASO
            
            avaliar(sol)
            melhorou ← VERDADEIRO
        FIM SE
    FIM ENQUANTO
    
    RETORNAR sol
FIM
```

### 3.2 Best Improving Local Search
```text
ALGORITMO: Busca_Local_Best_Improving(sol)
ENTRADA: sol (solução inicial)
SAÍDA: sol (solução melhorada)

INÍCIO
    melhorou ← VERDADEIRO
    
    ENQUANTO melhorou FAÇA
        melhorou ← FALSO
        melhor_movimento ← NULO
        melhor_delta ← 0.0
        
        CL_atual ← {i : i ∉ sol}  // Candidatos não na solução
        
        // AVALIAR INSERÇÕES
        PARA CADA cand_in ∈ CL_atual FAÇA
            delta_custo ← avaliar_custo_inserção(cand_in, sol)
            
            SE delta_custo > melhor_delta ENTÃO
                melhor_delta ← delta_custo
                melhor_movimento ← ("inserir", cand_in, NULO)
            FIM SE
        FIM PARA
        
        // AVALIAR REMOÇÕES
        PARA CADA cand_out ∈ sol FAÇA
            delta_custo ← avaliar_custo_remoção(cand_out, sol)
            
            SE delta_custo > melhor_delta ENTÃO
                melhor_delta ← delta_custo
                melhor_movimento ← ("remover", NULO, cand_out)
            FIM SE
        FIM PARA
        
        // AVALIAR TROCAS
        PARA CADA cand_in ∈ CL_atual FAÇA
            PARA CADA cand_out ∈ sol FAÇA
                delta_custo ← avaliar_custo_troca(cand_in, cand_out, sol)
                
                SE delta_custo > melhor_delta ENTÃO
                    melhor_delta ← delta_custo
                    melhor_movimento ← ("trocar", cand_in, cand_out)
                FIM SE
            FIM PARA
        FIM PARA
        
        // APLICAR MELHOR MOVIMENTO
        SE melhor_movimento ≠ NULO E melhor_delta > 1e-10 ENTÃO
            (tipo, cand_in, cand_out) ← melhor_movimento
            
            CASO tipo DE
                "inserir": sol ← sol ∪ {cand_in}
                "remover": sol ← sol \ {cand_out}
                "trocar": 
                    sol ← sol \ {cand_out}
                    sol ← sol ∪ {cand_in}
            FIM CASO
            
            avaliar(sol)
            melhorou ← VERDADEIRO
        FIM SE
    FIM ENQUANTO
    
    RETORNAR sol
FIM
```

## 4. Atualização da Lista de Candidatos (CL)
```text
ALGORITMO: Atualizar_CL(sol)
ENTRADA: sol (solução atual)
SAÍDA: CL atualizada (efeito colateral)

INÍCIO
    SE é_factível(sol) ENTÃO
        // Se já factível, qualquer subconjunto não na solução pode ser candidato
        CL ← {i : i ∉ sol, i ∈ {0,1,...,n-1}}
    SENÃO
        // Apenas subconjuntos que cobrem variáveis descobertas
        variáveis_descobertas ← obter_variáveis_descobertas(sol)
        CL ← ∅
        
        PARA i ← 0 ATÉ n-1 FAÇA
            SE i ∉ sol ENTÃO
                cobertura_subconjunto ← subconjuntos[i]
                SE variáveis_descobertas ∩ cobertura_subconjunto ≠ ∅ ENTÃO
                    CL ← CL ∪ {i}
                FIM SE
            FIM SE
        FIM PARA
    FIM SE
FIM
```

## 5. Algoritmo Principal GRASP
```text
ALGORITMO: GRASP_Principal(alpha, iterações, arquivo)
ENTRADA: alpha, iterações, arquivo
SAÍDA: melhor_solução

INÍCIO
    melhor_solução ← criar_solução_vazia()
    
    PARA i ← 1 ATÉ iterações FAÇA
        // FASE CONSTRUTIVA
        sol ← heurística_construtiva(alpha)
        
        // Garantir factibilidade
        SE NÃO é_factível(sol) ENTÃO
            tornar_factível(sol)
        FIM SE
        
        // FASE DE BUSCA LOCAL
        sol ← busca_local(sol)
        
        // ATUALIZAR MELHOR SOLUÇÃO
        SE sol.custo > melhor_solução.custo ENTÃO
            melhor_solução ← cópia(sol)
            
            SE verbose ENTÃO
                IMPRIMIR "(Iter. " + i + ") MelhorSol = " + melhor_solução
            FIM SE
        FIM SE
    FIM PARA
    
    RETORNAR melhor_solução
FIM
```
