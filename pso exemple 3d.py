import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Import pour la visualisation 3D

class Particle:
    def __init__(self, dim, bounds):
        """
        Initialisation d'une particule.
        :param dim: Dimension de l'espace du problème.
        :param bounds: Limites pour chaque dimension (tableau de forme [dim, 2]).
        """
        self.position = np.random.uniform(bounds[:, 0], bounds[:, 1], dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')  # Meilleur score atteint par la particule
        self.score = float('inf')  # Score actuel de la particule

def fitness_function(x):
    """
    Fonction objectif à minimiser.
    Exemple : Distance au point [3, 2, 1] dans un espace 3D.
    """
    return (x[0] - 3)**2 + (x[1] - 2)**2 + (x[2] - 1)**2

def is_feasible(position, constraints):
    """
    Vérifie si une position satisfait toutes les contraintes.
    :param position: Position à vérifier.
    :param constraints: Liste de fonctions de contraintes.
    """
    return all(constraint(position) for constraint in constraints)

def update_inertia(iteration, max_iter, inertia_start=0.9, inertia_end=0.4):
    """
    Met à jour le facteur d'inertie pour améliorer la convergence.
    :param iteration: Numéro de l'itération actuelle.
    :param max_iter: Nombre total d'itérations.
    :param inertia_start: Valeur initiale de l'inertie.
    :param inertia_end: Valeur finale de l'inertie.
    """
    return inertia_start - ((inertia_start - inertia_end) * (iteration / max_iter))

def pso_with_constraints(dim, bounds, constraints, num_particles=30, max_iter=100,
                          inertia_start=0.9, inertia_end=0.4, cognitive=1.5, social=1.5):
    """
    Implémentation de l'algorithme PSO avec gestion des contraintes.
    :param dim: Dimension de l'espace du problème.
    :param bounds: Limites pour chaque dimension (tableau de forme [dim, 2]).
    :param constraints: Liste de fonctions de contraintes.
    :param num_particles: Nombre de particules dans l'essaim.
    :param max_iter: Nombre maximum d'itérations.
    :param inertia_start: Inertie initiale.
    :param inertia_end: Inertie finale.
    :param cognitive: Facteur d'attraction vers le meilleur personnel.
    :param social: Facteur d'attraction vers le meilleur global.
    """
    # Initialisation des particules
    particles = [Particle(dim, bounds) for _ in range(num_particles)]
    global_best_position = np.zeros(dim)
    global_best_score = float('inf')
    convergence = []
    positions_history = []

    for iteration in range(max_iter):
        current_positions = []
        inertia = update_inertia(iteration, max_iter, inertia_start, inertia_end)

        for particle in particles:
            # Évaluation de la fonction objectif si la position est faisable
            if is_feasible(particle.position, constraints):
                particle.score = fitness_function(particle.position)

                # Mise à jour du meilleur personnel
                if particle.score < particle.best_score:
                    particle.best_score = particle.score
                    particle.best_position = np.copy(particle.position)

                # Mise à jour du meilleur global
                if particle.score < global_best_score:
                    global_best_score = particle.score
                    global_best_position = np.copy(particle.position)

            current_positions.append(particle.position)

        for particle in particles:
            # Mise à jour de la vitesse et de la position
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            cognitive_component = cognitive * r1 * (particle.best_position - particle.position)
            social_component = social * r2 * (global_best_position - particle.position)
            particle.velocity = inertia * particle.velocity + cognitive_component + social_component
            particle.position += particle.velocity

            # Respect des limites
            particle.position = np.clip(particle.position, bounds[:, 0], bounds[:, 1])

        convergence.append(global_best_score)
        positions_history.append(np.array(current_positions))
        print(f"Iteration {iteration+1}/{max_iter}, Best Score: {global_best_score:.6f}")

    return global_best_position, global_best_score, convergence, positions_history

# Définition du problème en 3D
dim = 3
bounds = np.array([[0, 10], [0, 10], [0, 10]])  # Limites pour les trois dimensions

def constraint1(x):
    # Exemple de contrainte : Somme des trois variables <= 20
    return np.sum(x) <= 20

def constraint2(x):
    # Exemple de contrainte : Toutes les variables doivent être non négatives
    return np.all(x >= 0)

constraints = [constraint1, constraint2]

# Exécution de l'algorithme PSO
best_position, best_score, convergence, positions_history = pso_with_constraints(
    dim, bounds, constraints)

print("\nMeilleure position trouvée :", best_position)
print("Meilleur score :", best_score)

# Visualisation des résultats en 3D
fig = plt.figure(figsize=(12, 6))

# Création d'un graphique 3D
ax = fig.add_subplot(121, projection='3d')
ax.set_xlim(bounds[0, 0], bounds[0, 1])
ax.set_ylim(bounds[1, 0], bounds[1, 1])
ax.set_zlim(bounds[2, 0], bounds[2, 1])
ax.set_title('Visualisation des Particules en 3D')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

particles_plot, = ax.plot([], [], [], 'ro', label='Particules')
best_plot, = ax.plot([], [], [], 'bo', label='Meilleure Position')
ax.legend()

# Graphique de convergence
ax2 = fig.add_subplot(122)
ax2.set_xlim(0, len(convergence))
ax2.set_ylim(0, max(convergence))
ax2.set_title('Convergence du PSO')
ax2.set_xlabel('Itération')
ax2.set_ylabel('Meilleur Score')
convergence_line, = ax2.plot([], [], 'r-')

# Fonction d'animation
def update(frame):
    positions = positions_history[frame]
    particles_plot.set_data(positions[:, 0], positions[:, 1])
    particles_plot.set_3d_properties(positions[:, 2])
    best_plot.set_data(best_position[0], best_position[1])
    best_plot.set_3d_properties(best_position[2])
    convergence_line.set_data(range(frame + 1), convergence[:frame + 1])
    return particles_plot, best_plot, convergence_line

ani = FuncAnimation(fig, update, frames=len(positions_history), interval=200, blit=True)

plt.show()
