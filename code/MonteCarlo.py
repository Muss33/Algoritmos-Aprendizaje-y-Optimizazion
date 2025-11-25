import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt


def iniciar(numero_entrenamientos, gamma, epsilon, dibujar=False, nombre_imagen="MonteCarlo_Resultado.png"):
    """
    Implementación del algoritmo Monte Carlo (First-Visit) para FrozenLake.

    A diferencia de Q-Learning o SARSA, Monte Carlo no actualiza la tabla paso a paso,
    sino que espera a que termine el episodio completo.
    Parámetros:
    - numero_entrenamientos: Número de episodios de entrenamiento.
    - gamma: Factor de descuento.
    - epsilon: Probabilidad de exploración (epsilon-greedy).
    - dibujar: Booleano para indicar si se debe dibujar el gráfico de recompensas.
    - nombre_imagen: Nombre del archivo para guardar el gráfico de recompensas.
    Retorna:
    - matrizQ: La tabla Q aprendida durante el entrenamiento.
    """
    env = gym.make("FrozenLake-v1", is_slippery=True)

    # Inicialización de Q(s,a) arbitraria (ceros)
    matrizQ = np.zeros((env.observation_space.n, env.action_space.n))

    # Para calcular el 'average(Returns)' necesitamos saber cuántas veces hemos visitado cada par (s,a)
    # y la suma de retornos acumulados. O podemos usar la media incremental.
    # Usaremos una matriz para contar N(s,a)
    N_visitas = np.zeros((env.observation_space.n, env.action_space.n))

    if dibujar:
        recompensa_episodio = np.zeros(numero_entrenamientos)

    # Loop para cada episodio
    for i in range(numero_entrenamientos):

        # 1. Generar un episodio completo siguiendo pi (policy)
        # En este caso pi es Epsilon-Greedy basada en Q
        episode_log = []  # Lista para guardar (Estado, Acción, Recompensa)

        estado = env.reset()[0]
        terminated, truncated = False, False

        # Generamos el episodio hasta el final
        while not (terminated or truncated):
            # Selección de acción (Epsilon-Greedy)
            if random.random() < epsilon:
                accion = env.action_space.sample()
            else:
                accion = np.argmax(matrizQ[estado, :])
                # Desempate aleatorio si todos son ceros (no sabe nada)
                if np.max(matrizQ[estado, :]) == 0:
                    accion = env.action_space.sample()

            # Ejecutar paso
            n_estado, reward, terminated, truncated, _ = env.step(accion)

            # Guardar en el log del episodio: St, At, Rt+1
            episode_log.append((estado, accion, reward))

            estado = n_estado

        # 2. Procesar el episodio
        G = 0  # El retorno acumulado

        # Recorremos el episodio desde el último paso hasta el primero (t = T-1, ... 0)
        for t in range(len(episode_log) - 1, -1, -1):
            st, at, rt = episode_log[t]

            # G <- gamma * G + R_(t+1)
            G = gamma * G + rt


            # "Unless the pair St, At appears in S0, A0... St-1, At-1"
            # Comprobamos si el par (st, at) apareció antes en este mismo episodio
            ha_aparecido_antes = False
            for k in range(t):
                if episode_log[k][0] == st and episode_log[k][1] == at:
                    ha_aparecido_antes = True
                    break

            if not ha_aparecido_antes:
                # Append G to Returns(St, At) y Q <- average(Returns)

                # NuevoPromedio = ViejoPromedio + (1/N) * (NuevoValor - ViejoPromedio)

                N_visitas[st, at] += 1
                alpha_mc = 1.0 / N_visitas[st, at]  # El 'alpha' real es 1/N para hacer la media exacta

                # Actualización
                matrizQ[st, at] = matrizQ[st, at] + alpha_mc * (G - matrizQ[st, at])

        if dibujar:
            if episode_log[-1][2] == 1:
                recompensa_episodio[i] = 1

    env.close()

    if dibujar:
        sum_rewards = np.zeros(numero_entrenamientos)
        window_size = 100
        for t in range(numero_entrenamientos):
            start = max(0, t - window_size + 1)
            sum_rewards[t] = np.mean(recompensa_episodio[start:(t + 1)])

        plt.figure(figsize=(10, 6))
        plt.plot(sum_rewards)
        plt.title('Monte Carlo (First-Visit) - Media Móvil de Recompensas')
        plt.xlabel('Episodio')
        plt.ylabel(f'Media de Recompensas (últimos {window_size})')
        plt.grid(True)
        plt.savefig(nombre_imagen)
        print(f"Gráfico guardado como: {nombre_imagen}")

    return matrizQ