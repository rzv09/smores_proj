import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from experiments.single_step.cross_region_federated_forgetting_fedprox import CrossRegionFederatedForgettingFedProx


def main():
    runs = 1
    experiment = CrossRegionFederatedForgettingFedProx('DO_allsites_allyears_20250611.csv', '3OEC_current_flow.csv', 2, 100, 0.01, '5min',
                                   12, 'mps')
    save_dir = f"./out/cross_region_federated_forgetting_fedprox_{runs}runs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment.run_experiment(runs, save_dir,  train_order=["SB1", "SB2", "SB3", "SB4", "FL1", "FL2", "FL3", "FL4"], early_stop=True, eval_all=False)

if __name__=='__main__':
    main()
