


class Particle:
    def __init__(self, dimensions):
        # Initialisation des coordonnées actuelles de la particule (c[]).
        # Ces coordonnées représentent la position actuelle dans l'espace de recherche.
        self.c = [0.0] * dimensions  # Initialisé à zéro, mais peut être modifié ultérieurement.

        # Initialisation des meilleures coordonnées trouvées par cette particule jusqu'à présent (cB[]).
        self.cB = [0.0] * dimensions

        # Initialisation de la vitesse de la particule (v[]), qui est utilisée pour déplacer la particule.
        self.v = [0.0] * dimensions

        # Initialisation de la valeur de fitness (ff), qui mesure la qualité de la position actuelle.
        # La valeur est fixée à la valeur minimale possible pour garantir qu'elle sera améliorée lors de l'optimisation.
        self.ff = -sys.float_info.max

        # Initialisation de la meilleure valeur de fitness atteinte par la particule (ffB).
        self.ffB = -sys.float_info.max

    def __repr__(self):
        """Méthode pour afficher les informations de la particule, utile pour le débogage."""
        return (f"Particle(c={self.c}, cB={self.cB}, v={self.v}, "
                f"ff={self.ff}, ffB={self.ffB})")
    


    import random

class Particle:
    def __init__(self, dimensions):
        self.c = [0.0] * dimensions  # Coordonnées actuelles de la particule
        self.cB = [0.0] * dimensions  # Meilleures coordonnées
        self.v = [0.0] * dimensions  # Vitesse de la particule
        self.ff = float('-inf')  # Valeur de la fonction de fitness actuelle
        self.ffB = float('-inf')  # Meilleure valeur de la fonction de fitness

class C_AO_PSO:
    def __init__(self):
        self.p = []  # Liste des particules
        self.rangeMax = []  # Plage maximale de recherche
        self.rangeMin = []  # Plage minimale de recherche
        self.rangeStep = []  # Pas de recherche
        self.cB = []  # Meilleures coordonnées globales
        self.ffB = float('-inf')  # Meilleure valeur de fitness globale

        # Paramètres de l'algorithme
        self.swarmSize = 0  # Taille de l'essaim
        self.parameters = 0  # Nombre de paramètres optimisés
        self.inertia = 0.0  # Inertie
        self.selfBoost = 0.0  # Boost personnel
        self.groupBoost = 0.0  # Boost collectif
        self.dwelling = False  # Si l'algorithme est en état de "dwelling" (considération)

    def InitPS(self, params, size, inertiaP, selfBoostP, groupBoostP):
        """Initialisation de l'algorithme PSO"""
        self.parameters = params
        self.swarmSize = size
        self.inertia = inertiaP
        self.selfBoost = selfBoostP
        self.groupBoost = groupBoostP

    def Preparation(self):
        """Préparation de l'essaim (initialisation des particules)"""
        self.GenerateRNDparticles()

    def Dwelling(self):
        """Logique de l'étape de 'dwelling' de l'algorithme"""
        self.dwelling = True
        self.ParticleMovement()

    def GenerateRNDparticles(self):
        """Générer les particules aléatoires"""
        self.p = [Particle(self.parameters) for _ in range(self.swarmSize)]
        for particle in self.p:
            particle.c = [self.RNDfromCI(self.rangeMin[i], self.rangeMax[i]) for i in range(self.parameters)]
            particle.v = [self.RNDfromCI(-abs(self.rangeMax[i]), abs(self.rangeMax[i])) for i in range(self.parameters)]

    def ParticleMovement(self):
        """Mettre à jour les positions et vitesses des particules"""
        for particle in self.p:
            for i in range(self.parameters):
                r1 = random.random()
                r2 = random.random()

                # Calcul de la vitesse (la mise à jour peut inclure les paramètres d'inertie, boost personnel et collectif)
                particle.v[i] = (self.inertia * particle.v[i] +
                                 self.selfBoost * r1 * (particle.cB[i] - particle.c[i]) +
                                 self.groupBoost * r2 * (self.cB[i] - particle.c[i]))

                # Mise à jour de la position
                particle.c[i] += particle.v[i]

            # Calcul de la fonction de fitness
            particle.ff = self.SeInDiSp(particle.c)  # Exemple de fonction de fitness

            # Mise à jour de la meilleure position et fitness de la particule
            if particle.ff > particle.ffB:
                particle.ffB = particle.ff
                particle.cB = particle.c.copy()

            # Mise à jour de la meilleure position globale
            if particle.ff > self.ffB:
                self.ffB = particle.ff
                self.cB = particle.c.copy()

    def SeInDiSp(self, coordinates):
        """Exemple d'une fonction d'évaluation (fitness function)"""
        # Cela peut être remplacé par une vraie fonction de fitness
        return sum(x**2 for x in coordinates)

    def RNDfromCI(self, min_value, max_value):
        """Générer une valeur aléatoire entre min_value et max_value"""
        return random.uniform(min_value, max_value)


class C_AO_PSO:
    def __init__(self):
        self.p = []  # Liste des particules
        self.rangeMax = []  # Plage maximale de recherche
        self.rangeMin = []  # Plage minimale de recherche
        self.rangeStep = []  # Pas de recherche
        self.cB = []  # Meilleures coordonnées globales
        self.ffB = float('-inf')  # Meilleure valeur de fitness globale

        # Paramètres de l'algorithme
        self.swarmSize = 0  # Taille de l'essaim
        self.parameters = 0  # Nombre de paramètres optimisés
        self.inertia = 0.0  # Inertie
        self.selfBoost = 0.0  # Boost personnel
        self.groupBoost = 0.0  # Boost collectif
        self.dwelling = False  # Si l'algorithme est en état de "dwelling" (considération)

    def InitPS(self, paramsP, sizeP, inertiaP, selfBoostP, groupBoostP):
        """Initialisation de l'algorithme PSO"""
        self.ffB = float('-inf')

        self.parameters = paramsP
        self.swarmSize = sizeP

        # Redimensionnement des plages et des particules
        self.rangeMax = [0.0] * self.parameters
        self.rangeMin = [0.0] * self.parameters
        self.rangeStep = [0.0] * self.parameters

        self.dwelling = False

        self.inertia = inertiaP
        self.selfBoost = selfBoostP
        self.groupBoost = groupBoostP

        self.p = [Particle(self.parameters) for _ in range(self.swarmSize)]

        for i in range(self.swarmSize):
            self.p[i].c = [0.0] * self.parameters  # Coordonnées de la particule
            self.p[i].cB = [0.0] * self.parameters  # Meilleures coordonnées
            self.p[i].v = [0.0] * self.parameters  # Vitesse de la particule

        self.cB = [0.0] * self.parameters  # Meilleures coordonnées globales



class C_AO_PSO:
    def __init__(self):
        # Initialisation des variables
        self.p = []  # Liste des particules
        self.rangeMax = []  # Plage maximale de recherche
        self.rangeMin = []  # Plage minimale de recherche
        self.rangeStep = []  # Pas de recherche
        self.cB = []  # Meilleures coordonnées globales
        self.ffB = float('-inf')  # Meilleure valeur de fitness globale
        self.swarmSize = 0  # Taille de l'essaim
        self.parameters = 0  # Nombre de paramètres optimisés
        self.inertia = 0.0  # Inertie
        self.selfBoost = 0.0  # Boost personnel
        self.groupBoost = 0.0  # Boost collectif
        self.dwelling = False  # Si l'algorithme est en état de "dwelling" (considération)

    def Preparation(self):
        """Préparation de l'algorithme : génération de particules ou mouvement des particules"""
        if not self.dwelling:
            self.ffB = float('-inf')  # Réinitialisation de la meilleure valeur de fitness
            self.GenerateRNDparticles()  # Génération des particules aléatoires
            self.dwelling = True  # Passage en mode "dwelling"
        else:
            self.ParticleMovement()  # Mouvement des particules

    def GenerateRNDparticles(self):
        """Génération de particules aléatoires"""
        # Implémente ici la logique pour générer des particules aléatoires
        print("Génération de particules aléatoires")

    def ParticleMovement(self):
        """Déplacement des particules"""
        # Implémente ici la logique pour le déplacement des particules
        print("Mouvement des particules")




import random

class C_AO_PSO:
    def __init__(self):
        self.p = []  # Liste des particules
        self.rangeMax = []  # Plage maximale de recherche
        self.rangeMin = []  # Plage minimale de recherche
        self.rangeStep = []  # Pas de recherche
        self.cB = []  # Meilleures coordonnées globales
        self.ffB = float('-inf')  # Meilleure valeur de fitness globale
        self.swarmSize = 0  # Taille de l'essaim
        self.parameters = 0  # Nombre de paramètres optimisés
        self.inertia = 0.0  # Inertie
        self.selfBoost = 0.0  # Boost personnel
        self.groupBoost = 0.0  # Boost collectif
        self.dwelling = False  # Si l'algorithme est en état de "dwelling" (considération)

    def GenerateRNDparticles(self):
        """Génération des particules aléatoires"""
        for s in range(self.swarmSize):
            for k in range(self.parameters):
                # Génération aléatoire des coordonnées
                self.p[s].c[k] = self.RNDfromCI(self.rangeMin[k], self.rangeMax[k])
                # Ajustement des coordonnées selon l'étape
                self.p[s].c[k] = self.SeInDiSp(self.p[s].c[k], self.rangeMin[k], self.rangeMax[k], self.rangeStep[k])
                # Initialisation des meilleures coordonnées
                self.p[s].cB[k] = self.p[s].c[k]
                # Initialisation de la vitesse
                self.p[s].v[k] = self.RNDfromCI(0.0, (self.rangeMax[k] - self.rangeMin[k]) * 0.5)

    def RNDfromCI(self, min_value, max_value):
        """Génération d'un nombre aléatoire entre min et max"""
        return random.uniform(min_value, max_value)

    def SeInDiSp(self, in_value, in_min, in_max, step):
        """Ajustement de la valeur en fonction de l'étape"""
        # Effectuer un ajustement en fonction du pas (step), ici un exemple simple
        return round(in_value / step) * step  # Rounding à l'échelon le plus proche





import random

class C_AO_PSO:
    def __init__(self):
        self.p = []  # Liste des particules
        self.rangeMax = []  # Plage maximale de recherche
        self.rangeMin = []  # Plage minimale de recherche
        self.rangeStep = []  # Pas de recherche
        self.cB = []  # Meilleures coordonnées globales
        self.ffB = float('-inf')  # Meilleure valeur de fitness globale
        self.swarmSize = 0  # Taille de l'essaim
        self.parameters = 0  # Nombre de paramètres optimisés
        self.inertia = 0.0  # Inertie
        self.selfBoost = 0.0  # Boost personnel
        self.groupBoost = 0.0  # Boost collectif
        self.dwelling = False  # Si l'algorithme est en état de "dwelling" (considération)

    def ParticleMovement(self):
        """Mouvement des particules"""
        for i in range(self.swarmSize):
            for k in range(self.parameters):
                # Composantes aléatoires du mouvement de la particule
                rp = self.RNDfromCI(0.0, 1.0)
                rg = self.RNDfromCI(0.0, 1.0)
                
                # Récupération des informations de la particule
                velocity = self.p[i].v[k]
                posit = self.p[i].c[k]
                positBest = self.p[i].cB[k]
                groupBest = self.cB[k]
                
                # Calcul de la nouvelle vitesse de la particule
                self.p[i].v[k] = (self.inertia * velocity + 
                                  self.selfBoost * rp * (positBest - posit) + 
                                  self.groupBoost * rg * (groupBest - posit))
                
                # Mise à jour de la position de la particule
                self.p[i].c[k] = posit + self.p[i].v[k]

                # Ajustement de la position selon l'étape
                self.p[i].c[k] = self.SeInDiSp(self.p[i].c[k], self.rangeMin[k], self.rangeMax[k], self.rangeStep[k])

    def RNDfromCI(self, min_value, max_value):
        """Génération d'un nombre aléatoire entre min et max"""
        return random.uniform(min_value, max_value)

    def SeInDiSp(self, in_value, in_min, in_max, step):
        """Ajustement de la valeur en fonction de l'étape"""
        return round(in_value / step) * step  # Rounding à l'échelon le plus proche







class C_AO_PSO:
    def __init__(self):
        self.p = []  # Liste des particules
        self.rangeMax = []  # Plage maximale de recherche
        self.rangeMin = []  # Plage minimale de recherche
        self.rangeStep = []  # Pas de recherche
        self.cB = []  # Meilleures coordonnées globales
        self.ffB = float('-inf')  # Meilleure valeur de fitness globale
        self.swarmSize = 0  # Taille de l'essaim
        self.parameters = 0  # Nombre de paramètres optimisés
        self.inertia = 0.0  # Inertie
        self.selfBoost = 0.0  # Boost personnel
        self.groupBoost = 0.0  # Boost collectif
        self.dwelling = False  # Si l'algorithme est en état de "dwelling" (considération)

    def Dwelling(self):
        """Mémorise les meilleures positions des particules"""
        for i in range(self.swarmSize):
            # Mémoriser la meilleure position pour la particule
            if self.p[i].ff > self.p[i].ffB:
                self.p[i].ffB = self.p[i].ff
                for k in range(self.parameters):
                    self.p[i].cB[k] = self.p[i].c[k]

            # Vérifier si la particule a une meilleure fitness globale
            if self.p[i].ff > self.ffB:
                self.ffB = self.p[i].ff
                for k in range(self.parameters):
                    self.cB[k] = self.p[i].c[k]




import math

class C_AO_PSO:
    def __init__(self):
        # Initialisation des variables comme avant
        pass

    def SeInDiSp(self, in_value, in_min, in_max, step):
        """Choix dans un espace discret"""
        if in_value <= in_min:
            return in_min
        if in_value >= in_max:
            return in_max
        if step == 0.0:
            return in_value
        else:
            return in_min + step * round((in_value - in_min) / step)



import random

class C_AO_PSO:
    def __init__(self):
        # Initialisation des variables comme avant
        pass

    def RNDfromCI(self, min_value, max_value):
        """Générateur de nombre aléatoire dans un intervalle personnalisé"""
        if min_value == max_value:
            return min_value
        
        Min = min(min_value, max_value)
        Max = max(min_value, max_value)
        
        return Min + (Max - Min) * random.random()





import random

class Particle:
    def __init__(self, parameters):
        self.parameters = parameters
        self.c = [random.uniform(-5, 5) for _ in range(parameters)]  # Coordonnées initiales
        self.v = [random.uniform(-5, 5) for _ in range(parameters)]  # Vitesses initiales

class C_AO_PSO:
    def __init__(self):
        self.parameters = 0
        self.swarmSize = 0
        self.inertia = 0
        self.selfBoost = 0
        self.groupBoost = 0
        self.p = []  # Liste des particules

    def InitPS(self, paramsP, sizeP, inertiaP, selfBoostP, groupBoostP):
        """Initialiser les paramètres de l'algorithme PSO"""
        self.parameters = paramsP
        self.swarmSize = sizeP
        self.inertia = inertiaP
        self.selfBoost = selfBoostP
        self.groupBoost = groupBoostP

        print(f"PSO initialisé avec {self.swarmSize} particules et {self.parameters} paramètres.")

    def GenerateRNDparticles(self):
        """Générer des particules aléatoires"""
        self.rangeMax = [5.0] * self.parameters  # Plage maximale pour chaque paramètre
        self.rangeMin = [-5.0] * self.parameters  # Plage minimale pour chaque paramètre

        # Création de l'essaim de particules
        self.p = [Particle(self.parameters) for _ in range(self.swarmSize)]

        for particle in self.p:
            particle.c = [random.uniform(self.rangeMin[i], self.rangeMax[i]) for i in range(self.parameters)]
            particle.v = [random.uniform(-abs(self.rangeMax[i]), abs(self.rangeMax[i])) for i in range(self.parameters)]

    def Preparation(self):
        """Préparer les particules"""
        self.GenerateRNDparticles()

    def RNDfromCI(self, min_value, max_value):
        """Générer une valeur aléatoire entre min_value et max_value"""
        return random.uniform(min_value, max_value)

# Fonction de test
def test_pso():
    # Initialiser l'algorithme PSO
    pso = C_AO_PSO()
    
    # Initialiser les paramètres de l'algorithme
    pso.InitPS(paramsP=3, sizeP=10, inertiaP=0.7, selfBoostP=1.5, groupBoostP=1.5)
    
    # Préparer l'essaim
    pso.Preparation()
    
    # Afficher l'état initial de l'essaim
    print("État initial de l'essaim :")
    for i, particle in enumerate(pso.p):
        print(f"Particule {i}: {particle.c}")

    # Effectuer plusieurs itérations (par exemple, 10)
    for epoch in range(10):
        print(f"\nÉpoque {epoch + 1}:")
        pso.Preparation()  # Mettre à jour l'état de l'essaim
        for i, particle in enumerate(pso.p):
            print(f"Particule {i}: {particle.c}")

# Tester l'algorithme PSO
test_pso()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Modify the Particle and PSO classes to store historical data
class Particle:
    def __init__(self, dimensions, min_val, max_val):
        self.c = np.random.uniform(min_val, max_val, dimensions)  # Position
        self.v = np.random.uniform(-1, 1, dimensions)  # Velocity
        self.ff = None  # Fitness
        self.cB = self.c.copy()  # Personal best position

    def evaluate(self, fitness_function):
        self.ff = fitness_function(self.c)

class PSO:
    def __init__(self, population_size, dimensions, min_val, max_val, w, c1, c2, fitness_function):
        self.population_size = population_size
        self.dimensions = dimensions
        self.min_val = min_val
        self.max_val = max_val
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.fitness_function = fitness_function

        self.particles = [Particle(dimensions, min_val, max_val) for _ in range(population_size)]
        self.gBest = self.particles[0]

        # Store history for visualization
        self.history_positions = []  # Positions of all particles per iteration
        self.history_fitness = []    # Global best fitness per iteration

    def update_velocity_and_position(self, particle):
        r1, r2 = np.random.rand(2)
        particle.v = (
            self.w * particle.v
            + self.c1 * r1 * (particle.cB - particle.c)
            + self.c2 * r2 * (self.gBest.c - particle.c)
        )
        particle.c = particle.c + particle.v
        particle.c = np.clip(particle.c, self.min_val, self.max_val)

    def optimize(self, iterations):
        for _ in range(iterations):
            for particle in self.particles:
                particle.evaluate(self.fitness_function)
                if particle.ff < self.fitness_function(particle.cB):
                    particle.cB = particle.c.copy()

                if particle.ff < self.gBest.ff:
                    self.gBest = particle

            self.history_positions.append([p.c.copy() for p in self.particles])
            self.history_fitness.append(self.gBest.ff)

            for particle in self.particles:
                self.update_velocity_and_position(particle)

        return self.gBest.c, self.gBest.ff

# Fitness function (example: sum of squares)
def fitness_function(x):
    return np.sum(x**2)

# Parameters
population_size = 30
dimensions = 3  # For 3D visualization
min_val = -10
max_val = 10
w = 0.5
c1 = 1.5
c2 = 1.5
iterations = 50

# Initialize and run PSO
pso = PSO(population_size, dimensions, min_val, max_val, w, c1, c2, fitness_function)
best_position, best_fitness = pso.optimize(iterations)

print(f"Best Position: {best_position}")
print(f"Best Fitness: {best_fitness}")

# Visualization: Swarm Effect in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for t, positions in enumerate(pso.history_positions):
    ax.clear()
    positions = np.array(positions)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color="blue", label="Particles")
    ax.scatter(pso.gBest.c[0], pso.gBest.c[1], pso.gBest.c[2], color="red", s=100, label="Global Best")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)
    ax.set_title(f"Swarm Effect (Iteration {t + 1})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.pause(0.1)

plt.show()

# Visualization: Convergence
plt.figure(figsize=(10, 6))
plt.plot(pso.history_fitness, marker="o", label="Global Best Fitness")
plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.title("Convergence of PSO")
plt.legend()
plt.grid()
plt.show()
