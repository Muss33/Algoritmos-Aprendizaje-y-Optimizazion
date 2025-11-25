import gymnasium as gym
import numpy as np
import random

from matplotlib import pyplot as plt


def iniciar(numero_entrenamientos, alpha, gamma, epsilon,dibujar=False,nombre_imagen="QLearning_FrozenLake_Resultado.png"):
    
    """
    Función para iniciar el algoritmo de Q-Learning en el entorno FrozenLake-v1 de Gymnasium.
    Parámetros:
    - numero_entrenamientos: Número de episodios de entrenamiento.
    - alpha: Tasa de aprendizaje.
    - gamma: Factor de descuento.
    - epsilon: Probabilidad de exploración (epsilon-greedy).
    - dibujar: Booleano para indicar si se debe dibujar el gráfico de recompensas.
    - nombre_imagen: Nombre del archivo para guardar el gráfico de recompensas.
    Retorna:
    - matrizQ: La tabla Q aprendida durante el entrenamiento.

    """

    env = gym.make("FrozenLake-v1", is_slippery=True)

    matrizQ = np.zeros((env.observation_space.n, env.action_space.n))

    if dibujar:
        recompensa_episodio=np.zeros(numero_entrenamientos)

    for i in range(numero_entrenamientos):

        # Obtenemos ObsType, que es la primera posición del agente en este caso
        # Env.reset() → tuple[ObsType, dict[str, Any]]  (Documentación gymnasium)
        estado_j = env.reset()[0]

        terminated, truncated = False, False
        """
        Texto extraído y adaptado de la documentación oficial de Gymnasium: 

        The episode ends if the following happens:

        terminated:
            The player moves into a hole.
            The player reaches the goal at max(nrow) * max(ncol) - 1 (location [max(nrow)-1, max(ncol)-1]).

        truncated:
            The length of the episode is 100 for FrozenLake4x4, 200 for FrozenLake8x8.
        """
        while not (terminated or truncated):
            # Acciones:
            # 0 izquierda; 1 abajo; 2 derecha; 3 arriba
            if random.random() < epsilon:
                action = env.action_space.sample()  # acción aleatoria
            else:
                action = np.argmax(matrizQ[estado_j, :])  # seleccionar acción con mayor valor (elige el índice de columna con mayor valor)
                #para evitar que cuando no sabe se mueva solo a la izquierda, hace aleatorio si no sabe nada
                if max(matrizQ[estado_j, :]) == 0:
                    action = env.action_space.sample()

            # Env.step(action: ActType) → tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]
            n_estado_j, reward, terminated, truncated, _ = env.step(action)

            # Modificamos valores de la tabla según Q-learning:
            # Q(St, At) ← Q(St, At) + alpha*[R(t+1) + gamma*max(Q(S(t+1), a)) - Q(St, At)]
            matrizQ[estado_j, action] = matrizQ[estado_j, action] + alpha * (
                        reward + gamma * max(matrizQ[n_estado_j, :]) - matrizQ[estado_j, action])

            estado_j = n_estado_j
        if dibujar:
            if reward == 1:
                recompensa_episodio[i] = 1

    env.close()

    if dibujar:
        # Trazar la media móvil de recompensas
        # Calcula la suma de recompensas de los últimos 100 episodios
        sum_rewards = np.zeros(numero_entrenamientos)
        window_size = 100
        for t in range(numero_entrenamientos):
            # Media de los episodios recientes para suavizar la curva de aprendizaje
            start = max(0, t - window_size + 1)
            sum_rewards[t] = np.mean(recompensa_episodio[start:(t + 1)])

        plt.figure(figsize=(10, 6))
        plt.plot(sum_rewards)
        plt.title('Q-Learning - Media Móvil de Recompensas (Ventana de 100)')
        plt.xlabel('Episodio')
        plt.ylabel(f'Media de Recompensas (de los últimos {window_size} episodios)')
        plt.grid(True)
        plt.savefig(nombre_imagen)
        print(f"Gráfico de recompensas guardado como: {nombre_imagen}")

    return matrizQ


