# Projeto de Machine Learning com MLflow

Este projeto é um pipeline completo de Machine Learning  com acompanhamento de experimentos usando **MLflow**. Inclui treinamento de modelos, registro de métricas e artefatos, e uma interface web para realizar previsões com o melhor modelo treinado. Todo o ambiente é containerizado com Docker.

---

## Tecnologias Utilizadas

- Python
- MLflow
- Scikit-Learn
- Pandas / Numpy
- Docker & Docker Compose
- Flask (para interface web)

---

## Como Executar o Projeto com Docker

### Pré-requisitos

Certifique-se de ter instalado:

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

---

### 1. Clone o repositório

```bash
git clone https://github.com/liana-cs/ConversionRate-AM.git
cd ConversionRate-AM
```

### 2. Suba os containers
```bash
docker-compose up --build
```
### 3. Acessar a Interface do MLflow
Abra o navegador em:
```bash
http://localhost:5000
```
Por fim, bastar subir um arquvio .csv para fazer o treinamento. Lá você pode acompanhar experimentos, métricas, parâmetros e artefatos salvos.
