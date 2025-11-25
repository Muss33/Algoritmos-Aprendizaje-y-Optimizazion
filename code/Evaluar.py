import numpy as np
import gymnasium as gym


def evaluate_policy(politica, n_eval_episodes=100, max_steps=100, genetico=False,ver_evaluacion=False):
    """
    Función para evaluar una política aprendida en el entorno FrozenLake-v1 de Gymnasium.
    Parámetros:
    - politica: La política a evaluar. Puede ser una matriz Q (para Q-Learning
        o SARSA) o un vector de acciones (para Algoritmo Genético).
    - n_eval_episodes: Número de episodios para la evaluación.
    - max_steps: Número máximo de pasos por episodio.
    - genetico: Booleano que indica si la política es del tipo Algoritmo Genético.
    - ver_evaluacion: Booleano para indicar si se debe renderizar el entorno durante la evaluación.
    Retorna:
    - mean_reward: Recompensa media obtenida durante los episodios de evaluación.
    - mean_steps: Número medio de pasos para alcanzar la meta en episodios exitosos.
    """
    #Segun si se quiere ver a la politica jugar o no
    if ver_evaluacion:
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode='human')
    else:
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)
    episode_rewards = []
    success_steps = []  # Lista para guardar pasos SOLO de episodios con éxito

    for episode in range(n_eval_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0  # Contador para este episodio
        done = False

        for step in range(max_steps):
            if genetico:
                # En el Genético, policy_input[state] ya es la acción (0,1,2,3)
                action = politica[state]
            else:
                # En Q-Learning/SARSA/MC, buscamos la acción con mayor valor Q
                action = np.argmax(politica[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            steps += 1  # Sumamos un paso
            state = new_state

            done = terminated or truncated
            if done:
                break

        episode_rewards.append(total_reward)

        # Solo añadimos los pasos a la lista si el episodio fue exitoso (llegó a la meta)
        if total_reward == 1:
            success_steps.append(steps)

    mean_reward = np.mean(episode_rewards)

    # Calculamos la media de pasos solo si hubo al menos un éxito
    if len(success_steps) > 0:
        mean_steps = np.mean(success_steps)
    else:
        mean_steps = 0.0

    return mean_reward, mean_steps