# Evolutionary Traveller (EvoTraveller)

<img src="./web/webpage.png" height=900>

## Description
This repository contains the Python code of a user-friendly visualization tool for solving simple examples of the Traveling Salesman Problem (TSP) using a Genetic Algorithm (GA). Ideal for students and educators interested in optimization and metaheuristics. This tool helps to understand the evolutionary strategy based on populations that allow find optimal solutions of complex combinatorial problems.
## Key Features
* Traveling Salesman Problem (TSP): is a classic optimization problem in which the goal is to find the shortest possible route that visits a set of given locations once and returns to the origin point [[Wikipedia](https://en.wikipedia.org/wiki/Travelling_salesman_problem)].
* Genetic Algorithm (GA): is type of metaheuristic inspired by the process of natural selection, which employs concepts like mutation, crossover, and selection to evolve a population of solutions towards optimal or near-optimal solution of a problem [[Wikipedia](https://en.wikipedia.org/wiki/Genetic_algorithm)].
* Interactive visualization: users can visually track the evolution of the solution over generations, observing how the GA converges to an optimal or near-optimal route.
* Customizable parameters: users can adjust genetic algorithm parameters such as population size, generations, tournament size, mutation, the percentage of elitism and crossover, and explore the impact of these parameters on the algorithm's performance.

## Deploying on Streamlit Cloud
This repository is deployed in Streamlit Cloud and can be run at: https://evotraveller.streamlit.app/

## Local installation
1. Clone the repository from GitHub.
2. Install the required dependencies listed in [requirements.txt](https://github.com/jismartin/evotraveller/blob/main/requirements.txt).
3. Run the Streamlit app using the command:
```
streamlit run main.py
```
## Code
* **algorithm.py** contains the functions of the genetic algorithm.
* **main.py** contains the code to generate the visualization page.

There is a **Jupyter notebook** with all the code that can be executed independently.

<!-- ga-series-links:start -->
## Teaching resources / Materiales docentes

[Portal en español](https://josemagalan.github.io/genetic-algorithms-lab/?lang=es) · [Portal in English](https://josemagalan.github.io/genetic-algorithms-lab/?lang=en)

| Resource / Recurso | Use or read / Usar o leer | Repository / Repositorio |
| --- | --- | --- |
| Portal / Índice de materiales | [Open / Abrir](https://josemagalan.github.io/genetic-algorithms-lab/) | [GitHub](https://github.com/josemagalan/genetic-algorithms-lab) |
| Representation / Representación | [Open / Abrir](https://josemagalan.github.io/representation-ga/) | [GitHub](https://github.com/josemagalan/representation-ga) |
| Initialisation and evaluation / Inicialización y evaluación | [Open / Abrir](https://josemagalan.github.io/initialization-and-evaluation-ga/) | [GitHub](https://github.com/josemagalan/initialization-and-evaluation-ga) |
| Selection / Selección | [Open / Abrir](https://josemagalan.github.io/selection-ga/) | [GitHub](https://github.com/josemagalan/selection-ga) |
| Crossover / Cruce | [Open / Abrir](https://josemagalan.github.io/crossover-ga/) | [GitHub](https://github.com/josemagalan/crossover-ga) |
| Mutation / Mutación | [Open / Abrir](https://josemagalan.github.io/mutation-ga/) | [GitHub](https://github.com/josemagalan/mutation-ga) |
| EvoTraveller | [Open / Abrir](https://evotraveller.streamlit.app/) | [GitHub](https://github.com/jismartin/evotraveller) |
| SelectionMechanisms | [Open / Abrir](https://josemagalan.shinyapps.io/SelectionMechanisms/) | [GitHub](https://github.com/josemagalan/SelectionMechanisms) |
| Learning handbook / Manual de aprendizaje | [Read / Leer](https://josemagalan.github.io/genetic-algorithms-lab-book/) | [GitHub](https://github.com/josemagalan/genetic-algorithms-lab-book) |

El manual está en construcción: su contenido y estructura pueden cambiar. / The handbook is a work in progress: its content and structure may change.
<!-- ga-series-links:end -->

## License
This project is licensed under the [MIT License](https://github.com/jismartin/evotraveller/blob/main/LICENSE).
