👤 Identificação: Manuela Menezes Alves

# Processo Seletivo – Intensivo Maker | AI

Pipeline completo de treinamento, conversão e otimização de uma CNN para classificação do dataset MNIST, com foco em deploy em dispositivos embarcados e IoT.

# Fluxo do Projeto
  - train_model.py   ->   model.h5 + model.keras
  - optimize_model.py   ->   model_dynamic.tflite + model.tflite

# Etapa 1 -- Treinamento (train_model.py)

  ## Dataset

    Foi utilizado o dataset MNIST, composto por 70.000 imagens em escala de cinza (28x28 pixels) de dígitos de 0 a 9.

    O dataset é carregado diretamente via TensorFlow/Keras, sem necessidade de download manual.

  ## Arquitetura do Modelo

    A arquitetura foi projetada com foco em eficiência computacional, priorizando uso em dispositivos com recursos limitados.

    ### Estrutura
      - 2 camadas convolucionais (Conv2D)
      - 2 camadas de pooling (MaxPooling)
      - 1 camada densa leve
      - Dropout para regularização
      Entrada (28x28x1)
      ↓
      Conv2D (16 filtros) + ReLU
      ↓
      MaxPooling
      ↓
      Conv2D (32 filtros) + ReLU
      ↓
      MaxPooling
      ↓
      Flatten
      ↓
      Dense (32) + ReLU
      ↓
      Dropout (0.3)
      ↓
      Dense (10) + Softmax

    ### Justificativa
      - Arquitetura enxuta -> menor consumo de memória e menor latência
      - Número reduzido de parâmetros -> ideal para Edge AI
      - Dropout -> reduz overfitting sem custo na inferência

  ## Treinamento

    - Otimizador: Adam
    - Função de perda: Sparse Categorical Crossentropy
    - Métrica: Accuracy
    - Épocas: 5
    - Batch size: 128

    ### Métricas
      O script reporta três métricas ao final do treino:
      - Accuracy (teste): fração de dígitos classificados corretamente -- métrica principal para datasets balanceados como o MNIST.
      - Loss (teste): valor da cross-entropy -- indica a confiança do modelo nas predições, não apenas se acertou ou errou.
      - Gap val/teste: diferença entre acurácia de validação (última época) e acurácia no conjunto de teste. Um gap baixo (< 2%) indica boa generalização e ausência de overfitting significativo.

    ### Resultado
      - Acurácia (teste): ~98%
      - Loss: baixo
      - Gap validação/teste: pequeno

  ## Salvamento
    O modelo é salvo em:
    - model.h5 -- formato legado, compatível com versões anteriores do Keras

# Etapa 2 -- Otimização (optimize_model.py)

  Foram aplicadas e comparadas duas técnicas de quantização para TensorFlow Lite.

  ## Técnica A -- Dynamic Range Quantization -> model_dynamic.tflite
  - Quantiza apenas os pesos de float32 para int8. As ativações permanecem em float32 durante a inferência. Não requer dataset de calibração — é a conversão mais simples e adequada para pipelines automatizados ou quando os dados de calibração não estão disponíveis.

  ## Técnica B — Full Integer Quantization → model.tflite
  - Quantiza pesos e ativações em int8 com base em um dataset de calibração (100 amostras do MNIST). O conversor mede os ranges reais de cada tensor e define fatores de escala (scale) e offset (zero-point) por camada, permitindo que toda a inferência ocorra em aritmética inteira.

  ## Comparativo e trade-offs

    Formato                           Tamanho         Redução vs .h5      Ativações quantizadas
    --------------------------------------------------------------------------------------------
    model.h5 (referência)              401.2KB            --               --
    model_dynamic.tflite (DRQ)          36.8KB            90.8%            Não
    model.tflite (Full Int8)            37.5KB            90.6%            Sim

    - Em modelos pequenos como este, os dois métodos produzem arquivos de tamanho similares em disco. A diferença entre as técnicas se manifesta na inferência em runtime: o Full Int8 elimina conversões float↔int entre camadas durante a execução, o que reduz latência e consumo energético em hardware com aceleração inteira (ARM Cortex-M, ESP32-S3). Para uma aplicação IoT realizando inferências contínuas com bateria, esse ganho energético é o fator determinante na escolha.
    - O model.tflite (Full Int8) é o artefato principal para deploy. O model_dynamic.tflite serve como baseline de comparação e opção para ambientes sem suporte completo a int8.

# Estrutura do Projeto

  ├── train_model.py          # Treinamento da CNN
  ├── optimize_model.py       # Conversão e quantização para TFLite
  ├── requirements.txt        # Dependências
  ├── model.h5                # Modelo Keras legado (gerado)
  ├── model.tflite            # Full Int8 -- deploy embarcado (gerado)
  └── README.md               # Este relatório
