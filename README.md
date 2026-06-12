# 🫁 Classificação de Patologias Pulmonares via Raios-X de Tórax com Deep Learning

Este repositório contém o código-fonte, dados de análise e documentação do projeto de Inteligência Artificial voltado à classificação automatizada de radiografias de tórax. O sistema utiliza uma **Rede Neural Convolucional (CNN)** customizada construída sobre o ecossistema TensorFlow/Keras para diagnosticar 6 condições pulmonares distintas.

O projeto cumpre com rigor os requisitos acadêmicos de complexidade estrutural, análise exploratória de dados e avaliação rigorosa através de matrizes de confusão em dados de teste inéditos.

---

## 📌 Visão Geral do Projeto

O objetivo principal deste trabalho é desenvolver e avaliar um modelo de aprendizado profundo capaz de auxiliar na triagem médica automatizada de exames de imagem. O modelo analisa radiografias e realiza a classificação multiclasse entre as seguintes condições:
1. **COVID-19**
2. **Enfisema Pulmonar (Emphysema)**
3. **Normal (Sem alterações)**
4. **Pneumonia Bacteriana**
5. **Pneumonia Viral**
6. **Tuberculose**

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

* **Linguagem:** Python 3.12
* **Deep Learning:** TensorFlow 2.x & Keras
* **Análise de Dados:** Pandas & NumPy
* **Visualização Gráfica:** Matplotlib & Seaborn
* **Métricas de Avaliação:** Scikit-Learn
* **Ambiente de Execução:** Google Colab (com aceleração de Hardware via **GPU T4**)
* **Armazenamento de Pesos:** Google Drive (integração via API nativa do Colab)

---

## 📊 Estrutura e Distribuição do Dataset

O conjunto de dados é composto por imagens de alta resolução organizadas nativamente nas pastas `train` (treino), `val` (validação) e `test` (teste). A volumetria observada no projeto compreende:

* **Imagens de Treino:** 14.551 imagens pertencentes às 6 classes.
* **Imagens de Validação:** 1.748 imagens pertencentes às 6 classes.
* **Imagens de Teste:** Utilizadas estritamente na avaliação final (gabarito cego).

### Distribuição Detalhada de Amostras por Classe (Treino):
* **Normal:** 2.871 imagens
* **COVID-19:** 2.417 imagens
* **Pneumonia Viral:** 2.413 imagens
* **Pneumonia Bacteriana:** 2.400 imagens
* **Enfisema (Emphysema):** 2.050 imagens
* **Tuberculose:** 287 imagens

O pipeline de dados conta com **Data Augmentation** em tempo de execução para o conjunto de treino, aplicando rotações (`rotation_range=20`), zoom (`zoom_range=0.15`), deslocamentos horizontais/verticais e espelhamento horizontal (`horizontal_flip=True`), prevenindo o superajuste (*overfitting*).

---

## 🏗️ Arquitetura da Rede Neural Convolucional (CNN)

O modelo foi projetado do zero visando extração progressiva de características hierárquicas. A rede possui **22.247.110 parâmetros totais**, divididos conforme a tabela estrutural abaixo:

| Tipo de Camada | Filtros / Unidades | Ativação | Detalhes Técnicos |
| :--- | :--- | :--- | :--- |
| **Conv2D (Entrada)** | 32 filtros (3x3) | ReLU | Input Shape: (224, 224, 3) |
| **BatchNormalization** | - | - | Estabilização do gradiente |
| **MaxPooling2D** | Pool (2x2) | - | Redução de dimensionalidade espacial |
| **Conv2D (Intermediária)** | 64 filtros (3x3) | ReLU | Extração de features de médio nível |
| **BatchNormalization** | - | - | Aceleração da convergência |
| **MaxPooling2D** | Pool (2x2) | - | Redução espacial |
| **Conv2D (Profunda)** | 128 filtros (3x3) | ReLU | Extração de features complexas |
| **BatchNormalization** | - | - | Redução de deslocamento interno |
| **MaxPooling2D** | Pool (2x2) | - | Redução espacial final |
| **Flatten** | 86.528 neurônios | - | Vetorização das features map |
| **Dense (Totalmente Conectada)**| 256 neurônios | ReLU | Processamento de alto nível |
| **Dropout** | Taxa de 0.5 (50%) | - | Regularização severa contra overfitting |
| **Dense (Saída)** | 6 neurônios | Softmax | Distribuição de probabilidade multiclasse |

---

## 📈 Pipeline de Treinamento e Otimização

O modelo foi compilado utilizando o otimizador **Adam**, função de perda **Categorical Crossentropy** e acompanhamento da métrica de acurácia. O treinamento utilizou duas estratégias críticas de automação:
1.  **Early Stopping:** Monitorou a perda de validação (`val_loss`), encerrando o processo na **Época 9** ao notar estagnação e reversão da curva de erro, evitando o desperdício de poder computacional e overfitting.
2.  **Model Checkpoint:** Salvou automaticamente em tempo real a melhor versão estável dos pesos diretamente no Google Drive (`melhor_modelo_raiox.keras`).

---

## 🔬 Resultados e Análise Crítica (Métricas de Teste)

A avaliação final do modelo foi gerada aplicando o gabarito nas imagens de teste nunca antes vistas pelo modelo, gerando a seguinte **Matriz de Confusão**:

* **Destaques Positivos (Altíssima Sensibilidade):** O modelo demonstrou performance excelente na detecção de **Tuberculose** (285 acertos corretos) e pulmões **Normais** (215 acertos corretos).
* **Desafio Clínico Esperado:** Houve uma sobreposição e confusão natural entre as classes de *Pneumonia Bacteriana* e *Pneumonia Viral* (121 falsos diagnósticos cruzados). Essa similaridade visual em exames radiológicos é um desafio documentado na literatura médica real.
* **Limitação Identificada (Enfisema):** O modelo tendeu a classificar incorretamente as amostras de Enfisema (Emphysema) direcionando-as majoritariamente para COVID-19 e Tuberculose. Cientificamente, essa discrepância fornece margem para discussões acadêmicas ricas sobre o desbalanceamento ou a necessidade de técnicas de *Transfer Learning* em trabalhos futuros.

---

## 📂 Como Executar este Repositório

### Pré-requisitos
1. Uma conta no Google com acesso ao Google Drive.
2. O arquivo compactado do dataset contendo as imagens nomeado como `archive.zip`.

### Instruções Passo a Passo
1.  Crie uma pasta no seu Google Drive chamada `colabIA`.
2.  Faça o upload do arquivo `archive.zip` dentro desta pasta `colabIA`.
3.  Abra o arquivo de notebook `.ipynb` fornecido neste repositório dentro do Google Colab.
4.  Ative a aceleração por hardware alterando o ambiente de execução para **GPU T4** (*Ambiente de Execução > Alterar o tipo de ambiente de execução > GPU T4*).
5.  Execute todas as células de código sequencialmente. O script realizará o mapeamento automático do Drive, extração veloz em memória virtual local e salvamento definitivo do modelo treinado.
