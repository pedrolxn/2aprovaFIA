## COMO EXECUTAR CADA ETAPA:

# Dados
Fonte: Portal de Dados Abertos do SUS — https://dadosabertos.saude.gov.br/dataset/arboviroses-dengue (recurso de 2026, arquivo DENGBR26.csv).<br><br>
Tamanho do arquivo usado: 470.451 registros, 121 colunas.

OBS: Um arquivo .zip foi disponibilizado nesse repositório (DENGBR26.csv.zip), com o arquivo .csv podendo ser obtido dentro dele, use-o como entrada dentro do Google Colab, antes de começar a execução, não esqueça de também importar o utils1e2.py para cada etapa, para a execução, o download dos notebooks também devem ser feitos (etapa1.ipynb)(etapa2.ipynb).

Etapa 1: Dentro do Google Colab, importar o arquivo /DENGBR26.csv, o caminho para execução dos dados já foi feito, basta importar. No canto esquerdo do ambiente, no botão escrito "arquivo". Logo após, basta clicar no botão também na parte superior da tela escrito "Executar tudo", as saídas aparecem logo abaixo de cada código, alguns trechos estão com anotações e existem textos de apoio para cada tópico.

Etapa 2: Também dentro do Google Colab, importar o arquivo /DENGBR26.csv, o caminho para execução dos dados já foi feito, basta importar. No canto esquerdo do ambiente, no botão escrito "arquivo". Logo após, basta clicar no botão também na parte superior da tela escrito "Executar tudo", as saídas aparecem logo abaixo de cada código, alguns trechos estão com anotações e existem textos de apoio para cada tópico.

Etapa 3: Dentro do Google Colab, abra o notebook etapa3.ipynb e importe o arquivo DENGBR26.csv, disponível dentro do ZIP deste repositório. O arquivo deve ficar no caminho /content/DENGBR26.csv. Depois, clique em "Ambiente de execução" e em "Executar tudo". O notebook utiliza os registros com hospitalização conhecida e realiza os cálculos da distribuição conjunta, probabilidades marginais e condicionais, regra da cadeia, Teorema de Bayes e independência condicional. Os resultados aparecem logo abaixo de cada código. Não é necessário importar o arquivo utils1e2.py para executar esta etapa.

Etapa 6: Dentro do Google Colab, abra o notebook `etapa6.ipynb` e certifique-se de importar o arquivo `DENGBR26.csv` (extraído do arquivo `.zip` disponibilizado no repositório) no caminho `/content/DENGBR26.csv`. Em seguida, vá em "Ambiente de execução" e clique em "Executar tudo". O notebook implementa um Sistema de Inferência Fuzzy (Lógica Difusa) do zero usando NumPy e Pandas, combinando as probabilidades estimadas pela Rede Bayesiana com sintomas clínicos para classificar o risco de hospitalização por dengue. O código gera automaticamente as métricas de desempenho (Acurácia, AUC, Log-loss) e os gráficos das funções de pertinência logo abaixo de cada célula. Não é necessário instalar bibliotecas externas adicionais.

** ADICIONEM O MANUAL DA PARTE DE VOCES **

