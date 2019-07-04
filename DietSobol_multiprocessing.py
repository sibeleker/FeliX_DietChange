import sys
import warnings
import numpy as np
import time
from ema_workbench.connectors.vensim import VensimModel
from ema_workbench import (MultiprocessingEvaluator,
                           TimeSeriesOutcome, 
                           perform_experiments,
                           RealParameter,
                           Constant, 
                           ema_logging, 
                           save_results)
import pandas as pd
from ema_workbench.em_framework.evaluators import LHS, SOBOL, MORRIS
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
from SALib.analyze import sobol

import seaborn as sns
import matplotlib.pyplot as plt
def plot_scores(scores, problem, n):
    scores_filtered = {k:scores[k] for k in ['ST','ST_conf','S1','S1_conf']}
    Si_df = pd.DataFrame(scores_filtered, index=problem['names'])
    Si_df.to_csv(directory+"SobolIndices_n{}.csv".format(n))

    sns.set_style('white')
    fig, ax = plt.subplots(1)

    indices = Si_df[['S1','ST']]
    err = Si_df[['S1_conf','ST_conf']]

    indices.plot.bar(yerr=err.values.T,ax=ax)
    fig.set_size_inches(12,8)
    fig.subplots_adjust(bottom=0.3)
    plt.savefig("Sobol_n{}_Emissions.png".format(n), dpi=600,  bbox_inches='tight')
    return fig




ema_logging.log_to_stderr(ema_logging.INFO)
ema_logging.log_to_stderr(ema_logging.DEBUG)

if __name__ == '__main__':
    directory = 'H:/MyDocuments/IIASA-Felix/Model files/Parallel_NatCom2/'

    vensimModel = VensimModel("FelixModel", wd=directory, model_file=r'FeliX3_Sibel_v16_NoExcel.vpm')
    
    #df_unc = pd.read_excel(directory+'Uncertainties_v3.xlsx', sheet_name='UncertaintiesNov2018')
    
    #vensimModel.outcomes = [TimeSeriesOutcome('Population')]
    df_unc = pd.read_excel(directory+'ScenarioFramework.xlsx', sheet_name='Uncertainties')
#     df_drop = pd.read_csv(directory+'toremove_n2500_sc0.csv')
#     df_drop.set_index('Parameter', inplace=True)
#     print(df_drop.index)
#     for p in df_drop.index:
#         df_unc = df_unc[df_unc['Uncertainties'] != p]
#     print(df_unc.shape)
    df_out = pd.read_excel(directory+'ScenarioFramework.xlsx', sheet_name='Outcomes')
    df_unc['Min'] = df_unc['Reference'] * 0.5
    df_unc['Max'] = df_unc['Reference'] * 1.5
    
    

    vensimModel.uncertainties = [RealParameter(row['Uncertainties'], row['Min'], row['Max']) for index, row in df_unc.iterrows()]
    
    #vensimModel.outcomes = [TimeSeriesOutcome(out) for out in df_out['Outcomes']]
    vensimModel.outcomes = [TimeSeriesOutcome('Total Agricultural and Land Use Emissions')]
    
    sc = 0
    n = 2500
    vensimModel.constants = [Constant('SA Diet Composition Switch', sc)]
    with MultiprocessingEvaluator(vensimModel, n_processes=7) as evaluator:
        for sc in [0, 2, 3, 4]:
            start = time.time()
            results_sa = evaluator.perform_experiments(n, uncertainty_sampling=SOBOL, reporting_interval=5000)
            end = time.time()
            print("Experiments took {} seconds, {} hours.".format(end-start, (end-start)/3600))
        
            fn = './Diet_Sobol_n{}_sc{}_v4_2050.tar.gz'.format(n, sc) #v2 is with narrow ranges for efficacy and removing some of the unimportant parameters
            #v3 is with the new multiplicative formulation, and new social norm parameters
            save_results(results_sa, fn)
    
#     
#     experiments, outcomes = results_sa
#     data = outcomes['Total Agricultural and Land Use Emissions'][:, -1]
#     problem = get_SALib_problem(vensimModel.uncertainties)
#     scores = sobol.analyze(problem, data, calc_second_order=True, print_to_console=True)
#     
#     plot_scores(scores, problem, n)
#     plt.show()
    



