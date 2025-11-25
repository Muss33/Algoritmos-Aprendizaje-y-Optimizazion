import numpy as np
import random
import time
import gymnasium as gym
import matplotlib.pyplot as plt


def run_episode(env, policy, episode_len=100):
    total_reward = 0
    obs, _ = env.reset()
    for t in range(episode_len):
        action = policy[obs]
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


# Cada política corre n_episodes veces y miras como de buena es (prob de llegar al final)
def evaluate_policy(env, policy, n_episodes=100):
    total_rewards = 0.0
    for _ in range(n_episodes):
        total_rewards += run_episode(env, policy)
    return total_rewards / n_episodes


# Genera política random
def gen_random_policy():
    #cadena de longitud 16 de valores entre 0 y 3 aleatorios (política aleatoria)
    return np.random.choice(4, size=((16)))


# Junta dos políticas (dos arrays de 16)
def crossover(policy1, policy2):
    new_policy = policy1.copy()
    for i in range(16):
        rand = np.random.uniform()
        if rand > 0.5:
            new_policy[i] = policy2[i]
    return new_policy


# Muta una política solo si tiene "suerte"
def mutation(policy, p=0.05):
    new_policy = policy.copy()
    for i in range(16):
        rand = np.random.uniform()
        if rand < p:
            new_policy[i] = np.random.choice(4)
    return new_policy


def alg_genetico(tamaño_poblacion: int, generaciones: int, prob_mutacion: float, elites: int, dibujar=False,
                 nombre_imagen="Genetico_FrozenLake_Resultado.png"):
    """
    Función para iniciar el Algoritmo Genético en el entorno FrozenLake-v1 de Gymnasium.
    Parámetros:
    - tamaño_poblacion: Número de políticas en cada generación.
    - generaciones: Número de generaciones a evolucionar.
    - prob_mutacion: Probabilidad de mutación para cada acción en una política.
    - elites: Número de mejores políticas que se mantienen sin cambios en cada generación.
    - dibujar: Booleano para indicar si se debe dibujar el gráfico de recompensas
    - nombre_imagen: Nombre del archivo para guardar el gráfico de recompensas.
    Retorna:
    - best_policy: La mejor política encontrada durante el entrenamiento.
    """
    
    env = gym.make('FrozenLake-v1', is_slippery=True, render_mode=None)

    ## Tamaño de cada poblacion de politicas
    n_policy = tamaño_poblacion

    # Cantidad de generaciones (bucle de reproducir, evaluar, generar, mutar)
    n_steps = generaciones

    # Parámetro que marca el numero de individuos que excluyes de ser cambiados por su valor tan alto
    elite_size = elites

    # Genera tamaño_poblacion politicas aleatorias
    policy_pop = [gen_random_policy() for _ in range(n_policy)]

    # Lista para guardar el mejor score de cada generación para la gráfica
    history_best_scores = []

    # Mientras queden generaciones por crear
    for idx in range(n_steps):
        # Lista de la puntuación de cada política (haces correr cada una 100 veces)
        policy_scores = [evaluate_policy(env, p) for p in policy_pop]

        # Guardar el mejor score de esta generación
        generation_max_score = max(policy_scores)
        history_best_scores.append(generation_max_score)

        # Lista que permite saber los indices de peor a mejor de los individuos

        policy_ranks = list(reversed(np.argsort(policy_scores)))
        # Se usa para sacar a los elites
        elite_set = [policy_pop[x] for x in policy_ranks[:elite_size]]

        # Comprobar si la suma de puntuaciones es 0 para evitar ZeroDivisionError
        sum_scores = np.sum(policy_scores)
        if sum_scores == 0:
            # Todas las políticas fallaron. Seleccionar padres uniformemente
            # para no crashear y seguir explorando.
            select_probs = np.ones(n_policy) / n_policy
        else:
            # Selección proporcional (ruleta)
            select_probs = np.array(policy_scores) / sum_scores

        # Te crea un array de tamaño (tam_población-elites) combinando los mejores individuos
        child_set = [crossover(
            policy_pop[np.random.choice(range(n_policy), p=select_probs)],
            policy_pop[np.random.choice(range(n_policy), p=select_probs)])
            for _ in range(n_policy - elite_size)]  # Usar elite_size

        # Cada hijo es sometido a una posible mutación
        mutated_list = [mutation(p, prob_mutacion) for p in child_set]
        # Se añaden los elites e hijos
        policy_pop = elite_set
        policy_pop += mutated_list

    policy_score = [evaluate_policy(env, p) for p in policy_pop]
    best_policy = policy_pop[np.argmax(policy_score)]

    env.close()

    # --- Generación del Gráfico ---
    if dibujar:
        plt.figure(figsize=(10, 6))
        plt.plot(history_best_scores, label='Mejor Individuo')
        plt.title('Algoritmo Genético - Evolución de la Tasa de Éxito')
        plt.xlabel('Generación')
        plt.ylabel('Tasa de Éxito (Probabilidad de Meta)')
        plt.grid(True)
        plt.legend()
        plt.savefig(nombre_imagen)
        print(f"Gráfico de evolución guardado como: {nombre_imagen}")

    return best_policy