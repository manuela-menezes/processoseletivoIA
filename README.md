<<<<<<< HEAD
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
=======
# Processo Seletivo – Intensivo Maker | AI

Bem-vindo(a) à **etapa prática do processo seletivo para o Intensivo Maker**.

Esta atividade tem como objetivo avaliar competências técnicas relacionadas a **Machine Learning**, **Visão Computacional** e **Otimização de modelos para sistemas embarcados (Edge AI)**, a partir da aplicação prática dos conhecimentos adquiridos nos cursos EAD da etapa anterior.

> 🎯 **Importante**  
> O foco deste desafio é avaliar sua capacidade de **projetar, treinar e otimizar um modelo de IA**.  

---

## 📌 Navegação Rápida

- 🏁 [Passo 0 – Antes de Tudo](#-passo-0-antes-de-tudo)
- ⚙ [Passo 1 – Preparando o Ambiente](#-passo-1-preparando-o-ambiente)
- 💻 [Passo 2 – O Desafio Técnico](#-passo-2-o-desafio-técnico)
  - 🎯 [Conjunto de Dados](#-conjunto-de-dados)
  - 📂 [Estrutura do Projeto](#-estrutura-do-projeto)
  - 📚 [Material de Apoio](#-material-de-apoio)
  - ⚖️ [Critérios de Avaliação](#️-critérios-de-avaliação)
- 📤 [Passo 3 – Instruções de Entrega](#-passo-3-instruções-de-entrega)
  - 📝 [Relatório do Candidato](#-relatório-do-candidato)

---

## 🏁 Passo 0: Antes de Tudo

Caso você **nunca tenha utilizado Git ou GitHub**, não se preocupe.  
Siga atentamente as etapas abaixo.


### 1️⃣ Criação de Conta no GitHub

1. Acesse: https://github.com  
2. Clique em **Sign up**  
3. Crie sua conta gratuita seguindo as instruções da plataforma  

(*O GitHub será utilizado para envio, versionamento e correção automática do seu projeto.*)


### 2️⃣ Instalação do Git

O **Git** é a ferramenta que permite versionar e enviar seu código para o GitHub.

- **Windows**  
  Baixe e instale o **Git Bash**:  
  https://git-scm.com/downloads

- **Linux / macOS**  
  Verifique se o Git já está instalado:
  ```bash
  git --version
  ```

---

## ⚙ Passo 1: Preparando o Ambiente

Para desenvolver o desafio, você deverá criar uma cópia deste repositório.

### 1️⃣ Fork do Repositório

<img width="219" height="45" alt="image" src="https://github.com/user-attachments/assets/5d629626-513a-445c-ba0f-e5bb3e225187" />

1. No canto superior direito desta página, clique em **Fork**  
2. Uma cópia deste repositório será criada no **seu perfil do GitHub**
(*O Fork permite que você trabalhe de forma independente sem alterar o repositório original.*)



### 2️⃣ Clone do Repositório

<img width="149" height="52" alt="image" src="https://github.com/user-attachments/assets/abbd331b-a005-4633-89c6-afd16acbe828" />

No repositório do **seu Fork**, clique em **<> Code**, copie a URL e execute:

```bash
git clone https://github.com/SEU_USUARIO/nome-do-repositorio.git
cd nome-do-repositorio
```
(*O comando `git clone` cria uma cópia do repositório.*)



### 3️⃣ Preparação do Ambiente de Execução

Você pode executar o projeto de **Três formas**. Escolha apenas uma.



#### Opção A – Ambiente Python Local 
Requisitos:
- Python **3.10 ou 3.11**
- pip

Instale as dependências com:

```bash
pip install -r requirements.txt
```



#### Opção B – Dev Container 
Este repositório inclui um **Dev Container** para facilitar a criação de um ambiente Python padronizado.

**Requisitos**
- VS Code
- Docker instalado
- Extensão **Dev Containers**

**Passos**
1. Abra o repositório no VS Code  
2. Selecione **“Reopen in Container”**  
3. Aguarde a criação automática do ambiente  

➡️ As dependências serão instaladas automaticamente.


#### Opção C - via browser
Você também pode abrir o container via github codespace

1. Clique em **<> Code**
2. Clique em **Codespaces**
3. Clique em **Create codespace on image**

<img width="482" height="436" alt="image" src="https://github.com/user-attachments/assets/37a1e99d-66d2-4730-b824-26f834bd8cc3" />


>  Será aberto uma instância do VS Code no seu navegador com o container configurado


---

## 💻 Passo 2: O Desafio Técnico

O desafio consiste em desenvolver um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos**, e posteriormente **otimizá-lo para execução em dispositivos Edge**, como sistemas embarcados e IoT.

O foco não é apenas obter alta acurácia, mas também **compreender o fluxo completo**:

**treinamento → salvamento → conversão → otimização**



### 🎯 Conjunto de Dados

Será utilizado o dataset **MNIST**, composto por imagens de dígitos manuscritos de **0 a 9**.
<img width="500" height="294" alt="image" src="https://github.com/user-attachments/assets/f323b4cc-d759-4e05-bb58-13e4d6dc7e5b" />

✔️ O dataset já está disponível na biblioteca **TensorFlow/Keras**, não sendo necessário download manual.

📌 *O MNIST é amplamente utilizado para introdução à Visão Computacional e Redes Neurais.*



###  ✅ Requisitos Obrigatórios

**Etapa 1:**  Treinamento do Modelo (`train_model.py`)

Implemente no arquivo `train_model.py` um código que realize:

- Carregamento do dataset MNIST via TensorFlow
- Construção e treinamento de um modelo de classificação baseado em **Redes Neurais Convolucionais (CNN)**  
  (utilizando camadas `Conv2D` e `MaxPooling`)
- Treinamento do modelo
- Exibição da **acurácia final** no terminal
- Salvamento do modelo treinado no formato **Keras** (`.h5`)

(*O modelo salvo será utilizado na etapa de otimização.*)



**Etapa 2:** Otimização do Modelo (`optimize_model.py`)

No arquivo `optimize_model.py`, implemente:

- Carregamento do modelo treinado
- Conversão para **TensorFlow Lite (`.tflite`)**
- Aplicação de técnica de otimização, como:
  - **Dynamic Range Quantization**

(**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de **Edge AI**.)



### 📂 Estrutura do Projeto

⚠️ **Atenção:**  
A estrutura e os nomes dos arquivos **não devem ser alterados**.

```plaintext
seu-repositorio/
├── .github/
│   └── workflows/
│       └── ci.yml            # 🤖 Pipeline de correção automática (NÃO ALTERAR)
├── .devcontainer/            # 🐳 Dev Container (opcional)
│   └── devcontainer.json
├── train_model.py            # ✏️ Treinamento do modelo
├── optimize_model.py         # ✏️ Conversão e otimização
├── requirements.txt          # 📄 Dependências do projeto
├── model.h5                  # 🤖 Modelo treinado (gerado)
├── model.tflite              # ⚡ Modelo otimizado (gerado)
└── README.md                 # 📝 Relatório final do candidato
```



### ⚠️ Restrições e Considerações de Engenharia

Este desafio é avaliado automaticamente por meio de um pipeline de
**integração contínua (CI)**, executado em um ambiente controlado e com
restrições de recursos computacionais.

Você **não precisa conhecer GitHub Actions** para realizar o desafio.
No entanto, é importante respeitar as diretrizes abaixo.

**Diretrizes para o Modelo**

- O modelo deve ser uma **CNN simples**, adequada para **Edge AI**
- Evite arquiteturas muito profundas ou complexas
- Recomenda-se utilizar **até 3 camadas convolucionais**
- **Não utilize modelos pré-treinados**
- Número de épocas **limitado** (ex: até 5)

#### Diretrizes de Execução

- Treinamento apenas em **CPU**
- Tempo total reduzido (compatível com CI)
- Código deve executar do início ao fim **sem intervenção manual**

> **Importante:**  
> O objetivo não é obter a maior acurácia possível, mas sim demonstrar
> **engenharia eficiente**, compatível com ambientes automatizados e
> restrições típicas de aplicações reais de Edge AI.



### 📚 Material de Apoio

Os cursos realizados na etapa anterior **devem ser utilizados como referência**.

- 📘 **Fundamentos de Inteligência Artificial para Sistemas Embarcados**
- 👁️ **Sistemas de Visão Computacional Embarcada**
- ⚙️ **Otimização de Modelos em Sistemas Embarcados**

(*Os exemplos apresentados nesses cursos podem ser adaptados e reutilizados neste desafio.*)



### ⚖️ Critérios de Avaliação

A avaliação considerará:

- **Funcionalidade**  
  Execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`

- **Edge AI**  
  Conversão correta para `.tflite` e aplicação de técnica de otimização

- **Documentação**  
  Preenchimento adequado do relatório (README.md)

---

## 📤 Passo 3: Instruções de Entrega

### ✔️ Validação 

Antes do envio, execute os scripts e confirme a geração dos arquivos:
- `model.h5`
- `model.tflite`



### ⬆️ Envio do Código

```bash
git add .
git commit -m "Entrega do desafio técnico - Seu Nome"
git push origin main
```



### 🔍 Verificação Automática

1. Acesse a aba **Actions** no GitHub  
2. Verifique se o workflow foi executado com sucesso (✅)  
3. Em caso de erro (❌), consulte os logs, corrija e envie novamente

<img width="807" height="363" alt="image" src="https://github.com/user-attachments/assets/d991d35b-2bc2-48f7-9ac7-cf5ca9dc452a" />



### 📎 Submissão Final

Copie o link do seu repositório e envie conforme orientações do processo seletivo no Moodle.

---

## 📝 Relatório do Candidato

O arquivo (`README.md`) deve ser utilizado como **relatório final do desafio**.

Preencha todas as seções de forma clara e objetiva.

> 💡 Dica: não é necessário um relatório extenso.  
> O mais importante é demonstrar **clareza nas decisões técnicas**.



**Exemplo:**

👤 Identificação: **Nome Completo:**


### 1️⃣ Resumo da Arquitetura do Modelo

Descreva, em palavras, a arquitetura da **CNN** implementada no arquivo
`train_model.py`.



### 2️⃣ Bibliotecas Utilizadas

Liste as principais bibliotecas utilizadas no projeto, preferencialmente
com suas versões.



### 3️⃣ Técnica de Otimização do Modelo

Explique qual técnica foi utilizada para otimizar o modelo no arquivo
`optimize_model.py`.



### 4️⃣ Resultados Obtidos

Informe o principal resultado obtido após o treinamento do modelo.



### 5️⃣ Comentários Adicionais (Opcional)

Utilize este espaço para comentar:
- Dificuldades encontradas  
- Decisões técnicas importantes  
- Limitações do modelo  
- Aprendizados durante o desafio


## 🆘 Suporte

Em caso de dúvidas:

- Consulte o material dos cursos EAD
- Leia atentamente este README
- Analise os logs das GitHub Actions
- Utilize os canais oficiais para contato com os instrutores

Boa sorte no processo seletivo.
****
>>>>>>> 899bbaa9c313b8c23de5f26b7e09e78f0ad4ebe5
