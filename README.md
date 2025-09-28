Markdown

# MVP: Manutenção Preditiva com Machine Learning para Ativos Industriais

**Autor:** Igor Cassimiro Assunção  
**Data:** 27/09/2025

---

## 1. Visão Geral do Projeto

Este projeto consiste em um MVP (Minimum Viable Product) de uma solução de Machine Learning para **manutenção preditiva**. O objetivo é prever falhas iminentes em componentes de máquinas industriais com 24 horas de antecedência, utilizando dados de telemetria e histórico de eventos. A solução permite que equipes de manutenção atuem de forma proativa, reduzindo custos com paradas não planejadas e otimizando a operação.

O problema foi modelado como uma **classificação binária supervisionada** em um dataset severamente desbalanceado.

## 2. Dataset Utilizado

O projeto utiliza o **Microsoft Azure Predictive Maintenance Dataset**, uma base de dados sintética que simula um ambiente industrial realista. Ela é composta por múltiplos arquivos, incluindo:
* **Telemetria (`telemetry.csv`):** Leituras horárias de sensores (voltagem, rotação, pressão, vibração).
* **Erros (`errors.csv`):** Histórico de erros ocorridos nas máquinas.
* **Manutenções (`maint.csv`):** Histórico de manutenções e trocas de componentes.
* **Falhas (`failures.csv`):** Registros de falhas de componentes (nosso alvo).
* **Máquinas (`machines.csv`):** Features estáticas de cada máquina (modelo e idade).

## 3. Metodologia

O fluxo de trabalho seguiu as melhores práticas de um projeto de Data Science, desde a análise até a avaliação final do modelo.

#### 3.1 Análise Exploratória de Dados (EDA)
A análise inicial revelou um **severo desbalanceamento de classes**, com os eventos de falha representando apenas ~2% do dataset. Isso norteou a escolha das métricas de avaliação e das estratégias de modelagem.

#### 3.2 Engenharia de Atributos
Foi a etapa mais crítica do projeto. Os dados brutos das diversas fontes foram transformados em um único dataset preditivo através de:
* **Criação de Features de Janela Temporal:** Cálculo de médias e desvios padrão móveis sobre uma janela de 24 horas para os dados de telemetria.
* **Integração de Dados de Eventos:** Utilização de `merge_asof` para enriquecer cada registro com o histórico mais recente de erros e manutenções.
* **Criação da Variável Alvo:** Definição de uma janela de 24 horas antes de cada falha real para rotular os dados como "Falha Iminente" (classe 1).

#### 3.3 Modelagem e Otimização
Foram comparados diversos modelos de classificação, incluindo `Logistic Regression`, `Random Forest` e `LightGBM`. O modelo proposto, um **`Specialist Custom LGBM`**, foi desenvolvido para otimizar o limiar de decisão, maximizando o F1-Score. A otimização de hiperparâmetros foi realizada com `GridSearchCV` e validação cruzada estratificada para garantir a robustez dos resultados.

## 4. Resultados Finais

O modelo final (`Specialist Custom LGBM` otimizado) demonstrou uma performance excelente no conjunto de teste, provando ser uma ferramenta valiosa para o negócio.

* **Recall (Taxa de Detecção):** **80%**
* **F1-Score (Equilíbrio P-R):** **72%**
* **Precision (Confiabilidade dos Alertas):** **65%**

Isso significa que o modelo foi capaz de **identificar corretamente 8 de cada 10 falhas reais** com 24 horas de antecedência.

#### Matriz de Confusão do Modelo Final
![Matriz de Confusão do Modelo Otimizado](image_64c08c.png)

## 5. Como Executar o Projeto

Este projeto está contido em um notebook do Google Colab (`.ipynb`) e é totalmente reprodutível.

#### 5.1 Pré-requisitos
* Uma conta Google para executar o Google Colab.
* (Opcional) Uma chave de API do Kaggle para baixar os dados diretamente, conforme instruído no notebook.

#### 5.2 Estrutura dos Arquivos
O repositório contém os seguintes artefatos principais:
.
├── predictive_maintenance_MVP.ipynb    # O notebook principal com todo o código
├── custom_classifier.py                # Módulo com a definição da classe do modelo customizado
├── modelo_final.pkl                    # O pipeline final treinado e salvo
└── README.md                           # Este arquivo


#### 5.3 Execução
1.  Abra o arquivo `predictive_maintenance_MVP.ipynb` no Google Colab.
2.  Execute as células em ordem. O notebook foi projetado para ser autossuficiente, instalando dependências e carregando os dados.
3.  A célula final do notebook carrega o `modelo_final.pkl` deste repositório e realiza uma predição no conjunto de teste para validar o funcionamento.

## 6. Próximos Passos
* **Refinar a Engenharia de Atributos:** Testar diferentes janelas de tempo (ex: 6h, 48h).
* **Explorar Modelos de Deep Learning:** Avaliar o uso de LSTMs ou Transformers para capturar padrões sequenciais mais complexos.
* **Implementar Métrica de Custo:** Otimizar o modelo com base em uma métrica de custo de negócio que penalize Falsos Negativos mais severamente.
