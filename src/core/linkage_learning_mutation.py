import random

class LinkageLearning:
    def __init__(self, instance):
        self.instance = instance

    def mutation_ll(self, parent):
        chrom_size = len(parent)

        xg = parent.copy()
        xh = parent.copy()
        xgh = parent.copy()

        xg_mutation = random.randint(0, chrom_size - 1)
        xh_mutation = random.randint(0, chrom_size - 1)
        while xh_mutation == xg_mutation:
            xh_mutation = random.randint(0, chrom_size - 1)

        fx = self.instance.fitness_function(parent)

        xg[xg_mutation] = not xg[xg_mutation]
        ind_xg = self.instance.generate_individual(xg)
        fxg = ind_xg.fitness
        df_g = abs(fxg - fx)
        if df_g > self.instance.EPSILON:
            self.instance.importance.add_importance(xg_mutation, df_g)

        xh[xh_mutation] = not xh[xh_mutation]
        ind_xh = self.instance.generate_individual(xh)
        fxh = ind_xh.fitness
        df_h = abs(fxh - fx)
        if df_h > self.instance.EPSILON:
            self.instance.importance.add_importance(xh_mutation, df_h)

        xgh[xg_mutation] = not xgh[xg_mutation]
        xgh[xh_mutation] = not xgh[xh_mutation]
        ind_xgh = self.instance.generate_individual(xgh)
        fxgh = ind_xgh.fitness
        df = abs(fxgh - fxh - fxg + fx)
        if df > self.instance.EPSILON:
            self.instance.evig.add_edge(xg_mutation, xh_mutation, df)

        if abs(fxgh - fxh) < self.instance.EPSILON:
            self.instance.importance.add_importance(xg_mutation, abs(fxgh - fxh))
        if abs(fxgh - fxg) < self.instance.EPSILON:
            self.instance.importance.add_importance(xh_mutation, abs(fxgh - fxg))

        return ind_xg, ind_xh, ind_xgh