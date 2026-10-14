# 📋 Checklist de Divergências: Tese (TCC.latex) vs. Pipeline Implementada

**Data da Auditoria:** 06 de Setembro de 2026  
**Documento Avaliado:** `TCC.latex` (724 linhas)  
**Base de Código Avaliada:** `train_timm_models.py`, `metrics.py`, `dataset.py`, `main.py`, `visualize.py`, `benchmark_scenarios.json`, `experiments/05_09_2026_run2/`

---

## 🎯 Sumário Executivo

A comparação detalhada entre a monografia (`TCC.latex`) e a base de código do projeto revela um contraste marcante:
- **O Capítulo 4 (Resultados) e o Capítulo 5 (Conclusão)** da tese estão **extremamente fiéis e sincronizados** com as execuções reais da pipeline (sessão `05_09_2026_run2`, sementes 42, 43 e 44, tabelas 4.1 e 4.2 com médias, desvios-padrão, intervalos de confiança e ablações).
- **Entretanto, o Resumo, a Introdução, os Objetivos e o Capítulo 3 (Metodologia)** contêm **divergências graves e contradições internas**, incluindo ferramentas que nunca foram usadas no código (como TensorFlow/Keras e TFLite INT8), protocolos simplistas (divisão 80/20 sem menção ao split em 3 vias com hash SHA256 anti-leakage) e omissão de componentes centrais da pipeline (Two-Stage Transfer Learning, Mixup, TTA, Temperature Scaling, LayerCAM e WeightedRandomSampler).

---

## 📊 Matriz Comparativa Rápida

| Componente / Tópico | Descrição no `TCC.latex` | Implementação Real na Pipeline | Status | Ação Recomendada |
| :--- | :--- | :--- | :---: | :--- |
| **Framework Deep Learning** | TensorFlow / Keras (Linha 450) | **PyTorch 2.x + timm** (PyTorch Image Models) | ❌ Divergência | Corrigir no LaTeX (Metodologia) |
| **Quantização & TFLite** | INT8 via TensorFlow Lite nos Objetivos (L. 273), Resumo (L. 143) e Metodologia (L. 489) | Não implementado (admitido como Trabalho Futuro nos Resultados L. 618 e Limitações L. 651) | ⚠️ Contradição Interna | Harmonizar o texto da tese ou implementar script |
| **Particionamento de Dados** | Divisão estratificada 80% Treino / 20% Validação (L. 481) | **Split em 3 Vias: 70% Treino / 10% Validação / 20% Teste**, agrupado por `lesion_id` com hash SHA256 anti-leakage | ❌ Divergência | Atualizar Metodologia para o split 70/10/20 agrupado |
| **Treinamento por Transferência** | Ajuste fino genérico (L. 468) | **Treinamento em 2 Estágios (Stage 1 Warmup + Stage 2 Deep Fine-Tuning)** com taxas diferenciais e CosineAnnealing | ⚠️ Omissão | Detalhar os dois estágios na Metodologia |
| **Critério de Early Stopping** | Monitoramento de perda de validação (*loss*) (L. 485) | Monitoramento de **`ham_val_mel_auc_roc`** (AUC-ROC de Melanoma) | ❌ Divergência | Corrigir critério para AUC-ROC na Metodologia |
| **Data Augmentation & Regularização** | Mutações geométricas básicas na classe minoritária (L. 458) | **Batch Mixup ($\alpha=0.2$)**, transformações afins completas e **Test-Time Augmentation (TTA)** na inferência | ⚠️ Omissão | Incluir Mixup e TTA na Metodologia |
| **Desbalanceamento de Classes** | Apenas Focal Loss (L. 473) | **WeightedRandomSampler ($1/\sqrt{N_c}$)** + Focal Loss ($\gamma=2.0, \alpha$), com `logit_adjust=0.0` | ⚠️ Incompleto | Descrever amostragem ponderada na Metodologia |
| **Calibração de Probabilidades** | Não mencionada na Metodologia (surge nos Resultados L. 566) | **Temperature Scaling ($T$)** pós-hoc via L-BFGS no HAM-val e cálculo de ECE com 15 bins | ⚠️ Omissão | Adicionar subseção de calibração na Metodologia |
| **Regras de Triagem e Limiares** | Abordagem descrita como puramente binária (L. 430) | **Dual-Perspective**: Classificador 7 classes + Limiares desacoplados (`sens90`, Youden, Malignancy Screening) | ⚠️ Omissão | Descrever regras operacionais na Metodologia |
| **Interpretabilidade (XAI)** | Não citada na Metodologia; chamada de Grad-CAM para todos (L. 620) | Grad-CAM (V1 a V4) e **Jacob Gil MSFA LayerCAM** com gradientes positivos para V5 | ⚠️ Incompleto | Adicionar subseção de XAI e especificar LayerCAM |

---

## 🔍 Detalhamento das Divergências e Contradições

### 1. Framework: PyTorch (`timm`) vs. TensorFlow / Keras
- **Trecho no LaTeX (Linha 450):**  
  > *"O desenvolvimento algorítmico será inteiramente implementado na linguagem Python, utilizando o framework TensorFlow/Keras pela sua compatibilidade madura com o ecossistema de dispositivos móveis através do TensorFlow Lite."*
- **Realidade no Código:** Toda a base (`train_timm_models.py`, `metrics.py`, `dataset.py`, `visualize.py`) utiliza estritamente `torch`, `torchvision` e `timm` (PyTorch Image Models). O modelo MobileNetV5 é importado diretamente de `mobilenetv5_300m.gemma3n` via `timm`.
- **Por que corrigir:** Uma banca examinadora invalidará a consistência do trabalho se a metodologia afirmar TensorFlow e o repositório for inteiramente em PyTorch.

---

### 2. A "Contradição do TensorFlow Lite e Latência INT8"
Existe uma cisão entre os capítulos iniciais e finais do próprio `TCC.latex`:
- **Nos Capítulos Iniciais (Resumo L. 143, Introdução L. 257, Objetivos Específicos L. 273, Metodologia L. 489-496):**
  - O texto promete que os modelos passaram por quantização INT8 via TensorFlow Lite e que a latência e o consumo de memória RAM foram mensurados.
- **Nos Capítulos Finais (Resultados L. 618, Limitações L. 651, Conclusão L. 669):**
  - O próprio texto admite:  
    > *"O número de parâmetros não determina sozinho tamanho quantizado, latência, memória de pico, consumo energético ou compatibilidade de operadores. Como essas dimensões não foram medidas, não se afirma que qualquer modelo esteja pronto para smartphone. Exportação, quantização e benchmark no dispositivo permanecem trabalhos futuros."* (Linha 618).  
    > *"Limitações: Não foram medidos latência, memória, consumo energético ou desempenho quantizado em dispositivo."* (Linha 651).
- **Decisão Necessária:**
  - **Opção A (Recomendada para o texto):** Adequar o Resumo, Introdução, Objetivo Específico 4 e Metodologia para afirmar que a análise computacional foca na *complexidade estrutural teórica (parâmetros e operações)* e na *resiliência diagnóstica cross-domain*, declarando expressamente a quantização INT8 e a aferição de latência em hardware físico como Trabalhos Futuros.
  - **Opção B (Se desejar manter a promessa no texto):** Criar um módulo na pipeline para exportação ONNX/PyTorch INT8 (ou conversão TFLite de V1-V4) e medir a latência em CPU/NPU simulada.

---

### 3. Divisão do HAM10000: Split 80/20 vs. Split em 3 Vias (70/10/20) com Hash SHA256
- **Trecho no LaTeX (Linha 481):**  
  > *"O dataset laboratorial primário (HAM10000) será submetido a um particionamento estratificado, dividindo-se as amostras em subconjuntos de treinamento e validação interna (na proporção de 80% e 20%, respectivamente)."*
- **Realidade no Código e nos Resultados (Linhas 506, 545):**  
  - A pipeline adota **70% Treino (7.012 imagens)**, **10% Validação de Calibração (1.008 imagens)** e **20% Teste In-Domain Intocado (1.995 imagens)**.
  - Mais importante: a divisão é **agrupada por `lesion_id`** (`grouped_stratified_split`), garantindo que múltiplas imagens da mesma lesão/paciente nunca estejam simultaneamente no treino e no teste (prevenção absoluta de *data leakage*).
  - O particionamento é auditado e gravado criptograficamente com hashes SHA256 em `split_manifest.json`.
  - Os próprios Resultados da tese citam o **HAM-val** (usado para calibração de temperatura e limiares) e o **HAM-test** (usado para reportar os resultados da Tabela 4.1).
- **Por que corrigir:** O split 3-way com agrupamento de lesão é um ponto fortíssimo de rigor acadêmico da pesquisa. A metodologia precisa descrevê-lo com orgulho, em vez de citar um split 80/20 genérico.

---

### 4. Protocolo de Treinamento em Dois Estágios (Two-Stage Transfer Learning)
- **Trecho no LaTeX (Linha 468):** Aborda o fine-tuning apenas conceitualmente.
- **Realidade no Código (`train_timm_models.py`):**
  - **Stage 1 (Linear Warmup):** 3 épocas com o backbone convolucional 100% congelado, treinando apenas a camada linear de projeção com learning rate mais alto ($1 \times 10^{-3}$ ou $5 \times 10^{-4}$).
  - **Stage 2 (Deep Fine-Tuning):** Descongelamento completo de todas as camadas, com learning rate reduzido ($1 \times 10^{-4}$ a $3 \times 10^{-5}$ dependendo do modelo), agendador `CosineAnnealingLR` e regularização de peso com AdamW.
- **Por que corrigir:** Essa mecânica de dois estágios é essencial para evitar a destruição catastrófica dos pesos pré-treinados no início da otimização.

---

### 5. Critério de Early Stopping e Seleção de Checkpoint
- **Trecho no LaTeX (Linha 485):**  
  > *"Este mecanismo monitorará continuamente a métrica de perda (loss) no conjunto de validação interna..."*
- **Realidade no Código:** A pipeline monitora explicitamente o **AUC-ROC de Melanoma na validação (`ham_val_mel_auc_roc`)** com tolerância delta (`selection_min_delta = 0.0005`).
- **Por que corrigir:** Em problemas com desbalanceamento severo (onde nevos benignos dominam 67% dos dados), a perda (*loss*) geral pode continuar diminuindo mesmo que o modelo esteja piorando na detecção de melanomas. Monitorar a AUC de melanoma alinha a engenharia de software ao objetivo clínico do TCC.

---

### 6. Técnicas de Data Augmentation, Mixup e TTA
- **Trecho no LaTeX (Linha 458):** Cita apenas transformações geométricas isoladas para a classe minoritária.
- **Realidade no Código:**
  - **Batch Mixup:** Aplicado nos lotes de treino com parâmetro de mistura $\alpha = 0.2$ ($Beta(\alpha, \alpha)$).
  - **Ablação na Tese:** A Tabela 4.2 e a Linha 604 do próprio LaTeX trazem a linha *"Sem Mixup"* (mostrando queda de 0,8865 para 0,8494 na AUC de melanoma no HAM-test).
  - **Test-Time Augmentation (TTA):** Inferência com 4 visualizações (imagem original, flip horizontal, flip vertical e ambos) para suavizar a variância estocástica.
- **Por que corrigir:** O leitor da tese chega na Tabela 4.2 dos Resultados e vê o "Mixup", mas não encontra nenhuma explicação prévia do que é o Mixup na Metodologia.

---

### 7. Calibração de Probabilidades (Temperature Scaling) e Limiares Operacionais
- **Trecho no LaTeX (Metodologia):** Não explica calibração nem a derivação dos limiares. Nos Resultados (Linhas 566-588), surge subitamente o parâmetro $T=0,6565$, o ECE e a política `sens90`.
- **Realidade no Código (`metrics.py` e `train_timm_models.py`):**
  - **Temperature Scaling:** Otimização pós-treino da temperatura $T$ sobre os logits da validação para minimizar a Negative Log-Likelihood (NLL) via otimizador L-BFGS.
  - **ECE (Expected Calibration Error):** Avaliado em 15 intervalos uniformes antes e depois da calibração.
  - **Limiar Clínico Desacoplado:** Seleção do limiar de probabilidade que atinge 90% de sensibilidade no HAM-val (`sens90`), transferido em regime zero-shot para o PAD-UFES-20.
  - **Mecanismo de Triagem Maligna Conjunta:** Regra combinada $P(\text{MEL}) + P(\text{BCC}) + P(\text{AKIEC}) \ge \tau_{\text{mal}}$.
- **Por que corrigir:** A modelagem matemática de calibração e limiares operacionais é um dos maiores pontos fortes do trabalho e deve estar formalizada na Metodologia.

---

### 8. Interpretabilidade Visual: Grad-CAM vs. LayerCAM no MobileNetV5
- **Trecho no LaTeX (Resultados L. 620):** Denomina todos os mapas de calor de "Grad-CAM".
- **Realidade no Código (`visualize.py`):**
  - Modelos V1 a V4 utilizam **Grad-CAM clássico**.
  - O MobileNetV5 (`mobilenetv5_300m.gemma3n`) utiliza **Jacob Gil MSFA LayerCAM**, interceptando as ativações em `raw_model.msfa.norm` e aplicando ponderação espacial elemento a elemento com gradientes positivos ($w_{ij}^k = \max(\partial Y^c/\partial A_{ij}^k, 0)$) para evitar os artefatos difusos de grade provocados pelos mixers de tokens e atenção do V5.
- **Por que corrigir:** Valoriza o trabalho demonstrando o domínio de técnicas avançadas de XAI adaptadas para arquiteturas híbridas modernas.

---

## 📝 Checklist de Ações para Sincronização Total

### 📌 Bloco A: Alterações no Documento da Tese (`TCC.latex`) — ✅ CONCLUÍDAS

- [x] **A.1 - Atualizar Framework (Capítulo 3 - Metodologia):**
  - Substituídas referências a `TensorFlow/Keras` por `Python 3.10+, PyTorch 2.x e a biblioteca timm (PyTorch Image Models)`.
  - Especificados os modelos exatos utilizados (`mobilenetv1_100`, `mobilenetv2_100`, `mobilenetv3_large_100`, `mobilenetv4_conv_large.e500_r256_in1k` e `mobilenetv5_300m.gemma3n`).
- [x] **A.2 - Resolver a Incoerência da Quantização e Latência:**
  - **No Resumo e Abstract:** Ajustado para focar no trade-off entre complexidade estrutural dos modelos e resiliência diagnóstica, posicionando a quantização e implantação como escopo futuro.
  - **Na Delimitação do Tema:** Substituída a menção a modelo quantizado em TensorFlow por complexidade estrutural de parâmetros, FLOPS e operadores para a borda.
  - **Nos Objetivos Específicos:** Readequado o item 4 para focar na comparação da complexidade estrutural de parâmetros e operadores.
  - **Na Metodologia (Seção 3.9):** Ajustada para "Considerações de Implantação e Quantização na Borda", alinhando formalmente com o Capítulo 4 (Resultados) e 5 (Conclusão).
- [x] **A.3 - Detalhar o Particionamento Zero-Leakage (Capítulo 3 - Metodologia):**
  - Substituído o split 80/20 pela divisão em 3 vias: **70% Treino (7.012 imagens)**, **10% Validação de Calibração (HAM-val, 1.008 imagens)** e **20% Teste In-Domain Intocado (HAM-test, 1.995 imagens)**.
  - Formalizado o agrupamento por `lesion_id` e a geração do manifesto criptográfico SHA-256 (`split_manifest.json`).
- [x] **A.4 - Inserir Subseção sobre Treinamento em Dois Estágios (Capítulo 3 - Metodologia):**
  - Descrito o **Estágio 1 (Linear Warmup)** com o extrator congelado (3 épocas) e o **Estágio 2 (Deep Fine-Tuning)** com taxas reduzidas e `CosineAnnealingLR`.
- [x] **A.5 - Corrigir o Critério de Early Stopping (Capítulo 3 - Metodologia):**
  - Atualizado para monitoramento do **AUC-ROC de Melanoma no HAM-val (`ham_val_mel_auc_roc`)** com tolerância mínima $\Delta = 0{,}0005$.
- [x] **A.6 - Descrever Data Augmentation Completo, Mixup e TTA (Capítulo 3 - Metodologia):**
  - Inserida a formulação matemática do **Batch Mixup** com $\alpha=0{,}2$ ($Beta(\alpha, \alpha)$).
  - Descrito o **Test-Time Augmentation (TTA)** com média das probabilidades de 4 visualizações.
- [x] **A.7 - Adicionar Subseção de Calibração de Probabilidades e Triagem (Capítulo 3 - Metodologia):**
  - Descrito o método de **Temperature Scaling ($T$)** minimizando NLL via L-BFGS no HAM-val.
  - Explicada a política de limiar operacional **`sens90`** e a Triagem Maligna Conjunta ($P(\text{MEL}) + P(\text{BCC}) + P(\text{AKIEC}) \ge \tau_{\text{mal}}$).
  - Formalizado o **ECE (Expected Calibration Error)** com 15 bins uniformes.
- [x] **A.8 - Adicionar Subseção de Métodos de Interpretabilidade Visual (Capítulo 3 e 4):**
  - Apresentada a formulação do **Grad-CAM** tradicional para V1–V4.
  - Apresentada a formulação do **MSFA LayerCAM** com gradientes positivos para o MobileNetV5 em `raw_model.msfa.norm`.
  - Atualizada a Seção 4.8 e a legenda da Figura 4.3 para identificar Grad-CAM (V4) e LayerCAM (V5).
- [x] **A.9 - Harmonizar a Taxonomia Multiclasse vs. Projeção Binária (Capítulo 3 - Metodologia):**
  - Detalhado o mapeamento de classes no PAD-UFES-20 (ACK+SCC $\to$ AKIEC, SEK $\to$ BKL, etc.) e a avaliação restrita às 5 classes compartilhadas sem penalização por DF e VASC.

---

### 📌 Bloco B: Ações e Melhorias na Base de Código (Pipeline)

- [ ] **B.1 - Limpeza do Repositório (`git_dirty`):**
  - O log e o texto da tese apontam `git_dirty: true` como limitação na reprodutibilidade. Realizar commit limpo de todos os scripts e arquivos de configuração.
- [ ] **B.2 - Documentação dos Hiperparâmetros:**
  - Garantir que os arquivos `benchmark_scenarios.json` e `RunConfig` em `main.py` possuam comentários explicativos sobre cada hiperparâmetro utilizado na monografia.
- [ ] **B.3 - (Opcional) Script de Exportação e Benchmark de Latência:**
  - Caso você deseje manter no TCC a promessa de medir latência e exportar para formato móvel, desenvolver um script (`export_edge.py`) que converta os modelos PyTorch treinados para ONNX / TorchScript / INT8 PTQ e compute a latência média de inferência (ms) em CPU.

---

## 🏆 Conclusão da Avaliação

O seu trabalho prático e experimental está em um **nível científico de excelência**. A inclusão de auditoria de dados contra vazamento (SHA256), a calibração com Temperature Scaling, o estudo rigoroso de ablação da V3, os intervalos de confiança com bootstrap clusterizado e a franqueza acadêmica ao discutir as limitações do domínio clínico colocam a pesquisa muito acima da média de um TCC de graduação.

O único ajuste crítico necessário é **alinhar o texto dos primeiros capítulos da monografia com a sofisticada engenharia que você realmente construiu no código**, eliminando menções anacrônicas ao TensorFlow/Keras e organizando a descrição metodológica para dar match total com o excelente Capítulo 4 de Resultados.
