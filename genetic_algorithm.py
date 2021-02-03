import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt

"""
Gene: a city (represented as (x,y) coordinates)
Individual (aka "chromosome"): a single route going through all cities
Population: a collection of possible routes (i.e, collection of individuals)
Parents: two routes that are to be combined to create a new route
"""


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Calculates distance between cities, using the pythagorean theorem
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))  # pythagorean theorem
        return distance

    # Cleaner way to output city coordinates
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


# The fitness is defined as the inverse of the route distance, since we want to minimize this distance.
# Large fitness score = low route distance
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]  # Here to make sure we start and end in the same city
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


# Create initial generation of individuals. An individual is a randomly ordered list of cities
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


# Creates initial population, as a list of individuals
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


# Returns a list of route IDs, which we can use to create the mating pool in the matingPool function
def selection(popRanked, eliteSize):
    selectionResults = []
    # In the 3 lines below, we set up the roulette wheel by calculating a relative fitness weight for each individual.
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


# Extracting the selected individuals from our population, and making a mating pool of that
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


# Ordered crossover where we randomly select a subset of the first parent string and fill the remainder of the route
# with genes from the second parent in the order in which they appear, without duplicating any genes
def crossover(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent2))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


# Generalize crossover function to create the offspring population
def crossoverPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):  # Using elitism to retain the best routes from the current population
        children.append(matingpool[i])

    for i in range(0, length):  # Function for filling out the rest of the next generation
        child = crossover(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children


# Mutation helps avoid local convergence by introducing novel routes that will allow us to explore other parts of the
# solution space. Similar to crossover, the TSP has a special consideration when it comes to mutation. Instead of having
# a low probability of a gene changing from 0 to 1, or vice versa, we'll use swap mutation so we don't drop any cities.
# Specifically, we'll have a specified low probability, of two cities swapping places in our route. This will be done
# for one individual in the mutation function.
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


# Extension of mutation function to run through the new population
def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


# Function for producing a new generation. Starts by ranking the routes in the current generation. We then determine our
# potential parents by running hte selection function, which allows us to create the mating pool. Finally, we create our
# new generation using the crossoverPopulation function, and the apply the mutations
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = crossoverPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


# Genetic algorithms
# def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
#     pop = initialPopulation(popSize, population)
#     print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
#
#     for i in range(0, generations):
#         pop = nextGeneration(pop, eliteSize, mutationRate)
#
#     print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
#     bestRouteIndex = rankRoutes(pop)[0][0]
#     bestRoute = pop[bestRouteIndex]
#     return bestRoute


# Plot the improvements and run GA
def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        print("Distance of generation" + str([i]) + ": " + str(1 / rankRoutes(pop)[0][1]))
        progress.append(1 / rankRoutes(pop)[0][1])

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


# Generate cities to travel between
cityList = []

for i in range(0,25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))


# Run the GA with the following parameters
geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)

