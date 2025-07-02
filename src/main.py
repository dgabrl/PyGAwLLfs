from src.core.gawll import GAwLL
from src.util.util import Util
from src.models.machine_learning_model import DT, KNN, MLP, RandomForest
from src.visualization.histogram_correlation import Histogram, Correlation
from src.visualization.graph import Graph
from src.visualization.values import SaveValues
import numpy as np

def model_training(dataset_type, model_name, X_trainset, d_trainset,
                   hidden_layer_sizes, learning_rate_init, max_iter,
                   min_samples_split_dt, k, max_depth, min_samples_split_rf):
    if model_name == 'dt':
        dt = DT(dataset_type=dataset_type, min_samples_split=min_samples_split_dt)
        dt.fit(X_trainset, d_trainset)
        model = dt

    elif model_name == 'knn':
        knn = KNN(dataset_type=dataset_type, k=k)
        knn.fit(X_trainset, d_trainset)
        model = knn

    elif model_name == 'mlp':
        mlp = MLP(
            dataset_type=dataset_type,
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
        )
        mlp.fit(X_trainset, d_trainset)
        model = mlp

    elif model_name == 'rf':
        rf = RandomForest(dataset_type=dataset_type, max_depth=max_depth,
                          min_samples_split=min_samples_split_rf)
        rf.fit(X_trainset, d_trainset)
        model = rf

    return model

def run_gawll(model, model_name,X_testset, d_testset, chrom_size):
    mutation_probability = 1.0 / chrom_size

    fitness_function = lambda chromosome: 0.98 * model.evaluate(
        X_testset, d_testset, dimensions=chromosome
    ) + 0.02 * (1 - np.mean(chromosome))

    instance = GAwLL(
        fitness_function=fitness_function,
        chrom_size=chrom_size,
        mutation_probability=mutation_probability,
        max_generations=max_generations,
    )
    print("Running GAwLL...\n")
    instance.run(42)
    instance.best_individual(model_name)
    print("Run finished\n")
    return np.array(instance.importance.get_importance()), instance.evig.interaction_matrix()

(dataset_type, chrom_size, X_trainset, d_trainset, X_testset, d_testset) = (
        Util.read_dataset('boson', perc_train=0.70)
)

variables = ['lepton_pT','lepton_eta','lepton_phi','missing_energy_magnitude','missing_energy_phi','jet1pt','jet1eta','jet1phi',
           'jet1b-tag','jet2pt','jet2eta','jet2phi','jet2b-tag','jet3pt','jet3eta','jet3phi','jet3b-tag','jet4pt','jet4eta',
           'jet4phi','jet4b-tag','m_jj','m_jjj','m_lv','m_jlv','m_bb','m_wbb','m_wwbb']


max_generations = 100
hidden_layer_sizes = (8,)
learning_rate_init = 0.1
max_iter = 500
min_samples_split_dt = 5
k = 3
max_depth = 10
min_samples_split_rf = 4
top_variables = 5

model_name = 'rf'

model = model_training(dataset_type, model_name, X_trainset, d_trainset,
                       hidden_layer_sizes, learning_rate_init, max_iter,
                       min_samples_split_dt, k, max_depth, min_samples_split_rf)

imp_gawll, int_matrix_gawll = run_gawll(model, model_name, X_testset, d_testset, chrom_size)

sv_instance = SaveValues()
sv_instance.save_importances(imp_gawll, f'{model_name}-GAwLL importances')
sv_instance.save_interaction_matrix(int_matrix_gawll, f'{model_name}-GAwLL interaction matrix')

graph_instance = Graph()
graph = graph_instance.graph(int_matrix_gawll, variables, model_name)
graph_instance.reduced_graph(graph, model_name)

histogram_instance = Histogram()
histogram_instance.histogram(variables, model_name, imp_gawll, None)