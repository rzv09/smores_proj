import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from experiments.single_step import federated_learning

def main():
    experiment = federated_learning.FederatedLearning('3OEC_current_flow.csv', 10, 10, 'avg', '5min',
                                   12, 'mps')
    save_dir = f"./out/Fl_100epochs_10runs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment.run_experiment(10, save_dir)

if __name__=='__main__':
    main()