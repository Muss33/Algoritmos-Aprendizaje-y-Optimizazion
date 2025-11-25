import QLearning, Sarsa, genetico, MonteCarlo, Evaluar
import time

#Algoritmo a ejecutar
QLearning_b=True
Sarsa_b=True
MonteCarlo_b=True
Genetico_b=True

#Parámetros para evaluar la política entrenada al acabar/crear gráfico del entreno
evaluar_politica = True
ver_evaluacion=True
dibujar=False

#Parámetros de los algoritmos
epsilon = 0.05
alpha = 0.1
gamma = 0.99
episodios = 20000

def main():
    if QLearning_b:
        start_time=time.time()
        print("Entrenant Q-Learning...")
        Q_table = QLearning.iniciar(
            numero_entrenamientos=episodios,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            dibujar=dibujar,
            nombre_imagen="QLearning_FrozenLake_Resultado.png"
        )
        end_time=time.time()
        print(f"Tiempo de entrenamiento (QLearning): {end_time - start_time:.4f} segundos")



    if Sarsa_b:
        print("Entrenant Sarsa...")
        start_time=time.time()
        Q_table = Sarsa.iniciar(
            numero_entrenamientos=episodios,
            alpha=alpha,  # Tasa de aprendizaje
            gamma=gamma,  # Factor de descuento
            epsilon=epsilon,
            dibujar=dibujar,
            nombre_imagen="Sarsa_FrozenLake_Resultado.png"
        )
        end_time=time.time()
        print(f"Tiempo de entrenamiento (Sarsa): {end_time - start_time:.4f} segundos")

    if MonteCarlo_b:
        print("Entrenant Monte Carlo...")
        start_time=time.time()

        Q_table=MonteCarlo.iniciar(
            numero_entrenamientos=episodios,
            gamma=gamma,  # Factor de descuento
            epsilon=epsilon,
            dibujar=dibujar,
            nombre_imagen="MonteCarlo_FrozenLake_Resultado.png"
        )
        end_time=time.time()
        print(f"Tiempo de entrenamiento (Monte Carlo): {end_time - start_time:.4f} segundos")


    if Genetico_b:
        print("Entrenant Genetic...")
        start_time=time.time()
        Q_table=genetico.alg_genetico(
            tamaño_poblacion=100,
            generaciones=30,
            prob_mutacion=0.05,
            elites=5,
            dibujar=dibujar,
            nombre_imagen="Genetico_FrozenLake_Resultado.png"
        )
        end_time=time.time()
        print(f"Tiempo de entrenamiento (Genético): {end_time - start_time:.4f} segundos")

    if evaluar_politica:
        mean_reward, mean_steps= Evaluar.evaluate_policy(Q_table,100,100,Genetico_b,ver_evaluacion)
        print(f"Tasa de éxito: {mean_reward * 100:.2f}%")
        print(f"Pasos promedio para llegar a la meta: {mean_steps:.2f}")

if __name__ == "__main__":
        main()
