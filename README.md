# Lab 10: Orquestração de Pipeline de IA - RAG, QLoRA e Otimização de Inferência

**Declaração de Integridade:** Este laboratório foi desenvolvido com auxílio de IA para estruturação de textos base, sendo totalmente revisado, testado e validado por **Alcimar Rosal**.

## 1. Resultados de Benchmark

Abaixo estão as métricas reais coletadas durante os testes realizados em ambiente Google Colab (GPU T4):

* **Carga do Modelo (QLoRA 4-bit):** Ocupação estática de **805.93 MB** de VRAM.
* **Teste de Estresse (Sem Otimização):** Falha Crítica. O sistema tentou alocar **30.91 GB** de VRAM para processar o contexto de 16k tokens, resultando em erro de **Out-Of-Memory (OOM)**.
* **Pipeline Final (KV Cache + SDPA):** Execução bem-sucedida com tempo de geração de **4.13 segundos** e pico de consumo de **3055.28 MB** de VRAM.

---

## 2. Parecer Técnico (Análise Arquitetural)

### Parte A: Mitigação do Colapso de VRAM
O erro de memória observado no Passo 3 é uma consequência direta da complexidade $O(n^2)$ do mecanismo de *Self-Attention* original. Sem otimização, a matriz de atenção tenta materializar todas as relações entre os 16 mil tokens simultaneamente, exigindo os ~31 GB reportados pelo sistema. 

A solução implementada utilizou três pilares: o **QLoRA** para manter os pesos do modelo comprimidos, o **KV Cache** para evitar o recálculo redundante de tokens já processados e o **SDPA (Scaled Dot Product Attention)**. Como a GPU T4 não suporta o FlashAttention-2 de forma nativa, o SDPA atuou como um fallback eficiente, realizando o cálculo da atenção por blocos (*tiling*). Isso impediu a explosão da memória e estabilizou o uso em cerca de 3 GB, tornando a inferência viável em hardware comercial.

### Parte B: Escalabilidade e State Space Models (SSM)
Mesmo com o uso de KV Cache e atenção otimizada, a arquitetura Transformer possui uma limitação física: o crescimento linear do cache de memória conforme o contexto aumenta. Se o requisito do cliente fosse de 2 milhões de tokens, o **KV Cache** ocuparia centenas de gigabytes, tornando o custo e a latência proibitivos para qualquer GPU atual. 

Para resolver esse gargalo, a indústria caminha para os **State Space Models (SSMs)**, como a arquitetura **Mamba**. Diferente dos Transformers, o Mamba não armazena o histórico completo de tokens; ele comprime as informações em um "estado oculto" de tamanho fixo. Isso resulta em uma complexidade de memória **$O(1)$**, permitindo o processamento de contextos massivos com consumo de VRAM constante, independente do tamanho da sequência.

## 3. Instruções de Execução

Para reproduzir este laboratório:
1. Abra o ficheiro `laboratorio_10.ipynb` presente neste repositório.
2. Abra o ficheiro no **Google Colab**.
3. Certifique-se de que o Ambiente de Execução está configurado para **Python 3 com GPU T4**.
4. Execute as células em ordem. O notebook está configurado para instalar automaticamente as bibliotecas `transformers`, `bitsandbytes` e `accelerate`.
---

## 4. Especificações Técnicas
* **Hardware:** NVIDIA T4 (15GB VRAM)
* **Modelo:** TinyLlama-1.1B-Chat-v1.0
* **Ambiente:** Google Colab / PyTorch 2.x
* **Versão de Entrega:** v1.0
