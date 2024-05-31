import numpy as np

mutants_color = (120, 110, 50)
class Population():
    def __init__(self, generator, n_bests = 0, n_crossover = 100, n_new = 0,
                 mutation_probability = 0.8, selection_pressure = 2, clone = True):
        self.n = n_bests * (1 + clone) + n_crossover + n_new
        self.n_bests = n_bests
        self.n_crossover = n_crossover
        self.n_new = n_new
        self.selection_pressure = selection_pressure
        self.clone = clone
        try:
            assert(self.n_new >= 0)
        except:
            raise Exception(f'''we have {- self.n_new} offsprings in excess for each generation, please adapt the 
            population parameters''')
        self.mutation_probability = mutation_probability
        self.generator = generator
        self.population = [self.generator() for i in range(self.n)]
    def evolve(self , n_bests = 20, mutationP = 0.7):
        ''' update the population with models + offsprings that are crossovers + some mutation with some probability
        cross over points are random, and offsprings have weights of the 1st parents at the left of the crossover point,
        and the weights from the other parent at the right.'''
        self.update_scores()
        self.update_probabilistic_perf()
        self.bests = self.deterministic_selection()
        if self.clone:
            self.clones = [b.copy() for b in self.bests]
            for c in self.clones:
                c.mutate()
                c.color = mutants_color
        else:
            self.clones = []
        for best_element in self.bests:
            best_element.color = (50, 50, 0)
        self.get_probabilistic_crossover_pop()
        self.new_ones = [self.generator() for i in range(self.n_new)]
        self.population = self.crossover_pop + self.new_ones + self.clones + self.bests
        assert(len(self.population) == self.n)
    def get_crossover_pop(self):
        '''performs crossover on the population'''
        self.crossover_pop = []
        couple_idxs = []
        for i in range(self.n_bests):
            for j in range(self.n_bests):
                couple_idxs.append(str(i)+'.'+str(j))
        parent_idxs = np.random.choice(couple_idxs, self.n - self.n_bests,replace=False)
        for i in range(self.n - self.n_bests - self.n_new):
            idx_1, idx_2 = [int(s) for s in parent_idxs[i].split('.')]
            child = self.population[idx_1].crossover(self.population[idx_2])
            if np.random.uniform(0,1,1) < self.mutation_probability:
                child.mutate()
            self.crossover_pop.append(child)
    def get_probabilistic_crossover_pop(self):
        '''performs crossover on the population'''
        self.crossover_pop = []
        for i in range(self.n_crossover):
            parent1, parent2 = np.random.choice(self.population, 2, replace = False, p = self.perf)
            child = parent1.crossover(parent2)
            if np.random.uniform(0,1,1) < self.mutation_probability:
                child.mutate()
            self.crossover_pop.append(child)
    def update_probabilistic_perf(self):
        f = self.scores - self.scores.min() + 0.01
        f = f / f.sum()
        f = f ** self.selection_pressure
        f = f / f.sum()
        self.perf = f
    def probabilistic_selection(self, p=2,replace=True):
        '''probabilist selection process, proportinal to performances minus the minimum performances, to the power of p, the selection pressure.
        (the higher p is, the more we select high performing solutions'''
        selected_population = np.random.choice(self.population, self.n_bests, replace=replace, p = f)
        return selected_population
    def deterministic_selection(self):
        '''deterministic selection process, we select the m best ones'''
        if self.n_bests > 0:
            ind = np.argpartition(self.scores, len(self.scores) - self.n_bests)[- self.n_bests:]
            selected_population = np.array(self.population)[ind]
            return selected_population.tolist()
        else:
            return []
    def reset(self):
        for x in self.population:
            x.reset()
    def update_scores(self):
        self.scores = np.array([pop_element.score for pop_element in self.population])
    def update_average_score(self):
        self.average_score = self.scores.mean()
    