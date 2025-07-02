import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import scipy.stats as stats

class Histogram:
  def __init__(self,output_dir='results'):
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    self.output_dir = output_dir

  def histogram(self,variables, model_name, importance_gawll, importance_model):

     if importance_gawll is not None:
        plt.figure(figsize=(12, 6))
        plt.bar(variables, importance_gawll)
        plt.title('GAwLL Importances by Variable')
        plt.xlabel('Variables')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{model_name}-GAwLL Histogram.png'))
        plt.close()
     else:
        print('GAwLL importances array is empty')

     if importance_model is not None:
        plt.figure(figsize=(12, 6))
        plt.bar(variables, importance_model)
        plt.title(f'{model_name} Importances by Variable')
        plt.xlabel('Variables')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{model_name} Histogram.png'))
        plt.close()
     else:
        print('Model importances array is empty')

class Correlation:
  def __init__(self,output_dir='results'):
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    self.output_dir = output_dir

  def correlation(self,model_name, importance_gawll, importance_model):
    if importance_gawll is None or importance_model is None:
       vectors = {'GAwLL': importance_gawll, 'Model': importance_model}
       print(f"{name} importances array is empty" for name, vector in vectors.items() if not vector)
    else:
       importance_gawll = np.array(importance_gawll)
       imp_gawll_norm = MinMaxScaler().fit_transform(importance_gawll.reshape(-1, 1)).flatten()
       imp_model_norm = MinMaxScaler().fit_transform(importance_model.reshape(-1, 1)).flatten()

       slope, intercept, r, p, stderr = stats.linregress(imp_model_norm, imp_gawll_norm)

       fig, ax = plt.subplots()
       ax.plot(imp_model_norm, imp_gawll_norm, linewidth=0, marker='s')
       line = f'r={r:.2f}'
       ax.plot(imp_model_norm, intercept + slope * imp_model_norm, label=line)

       ax.set_xlabel(f"{model_name}")
       ax.set_ylabel('GAwLL')
       ax.legend(facecolor='white')
       plt.savefig(os.path.join(self.output_dir, f'{model_name}-Correlation.png'))
       plt.close()

class CorrelationMatrix:
  def __init__(self,output_dir = 'results'):
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    self.output_dir = output_dir

  def correlation_matrix(self,dt_perm_imp,knn_perm_imp,mlp_perm_imp,rf_perm_imp,
                        gawll_dt_imp,gawll_knn_imp,gawll_mlp_imp,gawll_rf_imp,
                        dt_intrinsic_imp,rf_intrinsic_imp):

    importances_per_model = [dt_intrinsic_imp, rf_intrinsic_imp,
                            dt_perm_imp, knn_perm_imp, mlp_perm_imp, rf_perm_imp,
                            gawll_dt_imp, gawll_knn_imp, gawll_mlp_imp, gawll_rf_imp]

    importances_names = ['DT', 'RF','DT-Perm', 'KNN-Perm', 'MLP-Perm', 'RF-Perm',
                         'DT-GAwLL','KNN-GAwLL', 'MLP-GAwLL', 'RF-GAwLL']
    vector_empty = False

    for vector in importances_per_model:
      if len(vector) == 0:
        vector_empty = True
        break

    if not vector_empty:
      correlations = np.corrcoef(np.array(importances_per_model))

      plt.figure(figsize=(9,9))
      sns.heatmap(correlations, annot=True, cmap="coolwarm",
                  xticklabels=importances_names, yticklabels=importances_names)
      plt.title('Correlations Between Importances')
      plt.savefig(os.path.join(self.output_dir,'Correlations Importances Matrix.png'))
      plt.close()
      print('Correlations Importances Matrix Saved')
    else:
      print('Some importance vector is empty')