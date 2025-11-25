# 🧠 Algoritmos de Aprendizaje e Optimización en Frozen Lake

[![Estado del Proyecto](https://img.shields.io/badge/Estado-Finalizado-success)](https://github.com/Muss33/Algoritmos-Aprendizaje-y-Optimizazion)
[![Lenguaje Principal](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Entorno RL](https://img.shields.io/badge/Gymnasium-FrozenLake-green)](https://gymnasium.farama.org/)

---

##  1. Resumen y Objetivos

Este proyecto representa una inmersión profunda en la implementación y el análisis comparativo de algoritmos fundamentales de la Inteligencia Artificial: **Aprendizaje por Refuerzo (RL)** y **Algoritmos Genéticos (AG)**.

El campo de batalla elegido es el entorno estocástico **Frozen Lake** de Gymnasium (OpenAI Gym), un escenario ideal para evaluar cómo diferentes estrategias de aprendizaje gestionan la incertidumbre.

### Objetivos Clave:

- **Implementación Fiel:** Desarrollar las versiones *on-policy* (SARSA) y *off-policy* (Q-Learning) para observar su comportamiento.
- **Análisis de Retornos:** Implementar Monte Carlo para comparar la actualización basada en el retorno final frente a la diferencia temporal (TD).
- **Exploración de Optimización:** Aplicar un Algoritmo Genético para contrastar el aprendizaje basado en recompensas con la optimización evolutiva de políticas.
- **Influencia de Hiperparámetros:** Determinar y documentar cómo los parámetros ($\alpha, \gamma, \epsilon$, tasa de mutación, etc.) afectan la convergencia y estabilidad de la solución.

---

##  2. El Entorno Frozen Lake

Frozen Lake es un juego de cuadrícula donde un agente (Silla) debe navegar hacia una meta (Regalo) evitando caer en agujeros (Lagos congelados).

| Símbolo | Significado |
| :--- | :--- |
| **S** | Inicio (Start) |
| **F** | Suelo Congelado (Frozen) |
| **H** | Agujero/Lagos (Hole) |
| **G** | Meta (Goal) |

> ** Característica Crítica: Estocasticidad**  
> La propiedad `is_slippery=True` introduce un desafío crucial: al intentar moverse en una dirección, el agente solo tiene una **probabilidad** de moverse en esa dirección y una probabilidad de deslizarse a una de las dos direcciones adyacentes. Esto obliga a los algoritmos a encontrar políticas robustas, no solo un camino fijo.

---

##  3. Algoritmos en Detalle

### Aprendizaje por Refuerzo (RL)

| Algoritmo | Mecanismo de Actualización | Foco Principal | Notas en el Informe |
| :--- | :--- | :--- | :--- |
| **Q-Learning** | Diferencia Temporal (TD) | **Explotación** (*Off-Policy*) | Se analiza su superioridad en la búsqueda del valor óptimo $Q^*$. |
| **SARSA** | Diferencia Temporal (TD) | **Exploración** (*On-Policy*) | Se compara su curva de aprendizaje más conservadora, siguiendo la política actual. |
| **Monte Carlo** | Retorno Total del Episodio | **Exploración** (Promedio) | Necesita episodios completos; se documenta su lenta convergencia inicial. |

### Algoritmo Genético (AG)

- **Representación:** Cada individuo es un array que codifica una política completa para el mapa ($16 \times 4$ estados).  
- **Función Fitness:** Evaluación basada en el porcentaje de éxitos en un número fijo de partidas.  
- **Mecanismos:** Se incluye la gestión de **individuos élite** para la preservación de las mejores soluciones.

---

##  4. Análisis de Resultados

El informe (`informe_practica_ia.pdf`) contiene una sección exhaustiva de resultados con:

### A. Gráficas de Rendimiento

Se muestran comparativas de la probabilidad de éxito promedio, destacando:

- La inestabilidad y el sobreaprendizaje con un $\alpha$ alto.  
- El efecto de la "miopía" del agente con un $\gamma$ bajo.  
- El balance entre $\epsilon$ y la capacidad de escapar de máximos locales.

### B. Comparativa Global

Se presenta una tabla y gráficas comparando:

- **Tasa de Éxito Media:** Rendimiento final de las políticas óptimas de cada algoritmo.  
- **Tiempo de Entrenamiento:** Análisis de la eficiencia temporal, donde el Algoritmo Genético se dispara debido a la complejidad de su función *fitness*.

> **Conclusión Clave del Informe:** El estudio demuestra que **Q-Learning** tiende a converger más rápido a una política *greedy* más efectiva, mientras que la baja escalabilidad del Algoritmo Genético lo limita severamente para problemas de mayor tamaño.

---

##  5. Estructura y Ejecución

###  Estructura del Repositorio

| Archivo/Directorio | Descripción |
| :--- | :--- |
| `MonteCarlo.py` | Implementación del algoritmo Monte Carlo. |
| `Sarsa.py` | Implementación del algoritmo SARSA. |
| `QLearning.py` | Implementación del algoritmo Q-Learning. |
| `genetico.py` | Implementación del Algoritmo Genético. |
| `Evaluar.py` | Script para evaluar políticas generadas por los algoritmos. |
| `__main__.py` | Script de ejecución principal. |



###  Requisitos y Ejecución

Para replicar los resultados, asegúrate de tener instalado **Python 3.x** y las siguientes librerías:

```
pip install numpy gymnasium matplotlib

```

Para la ejecución del código, descárgate el repositorio y ejecuta el archivo ```__main__.py```


Autores: Alejandro Martínez Hermosa, Martín Serra Rubio.
