import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from experiments.single_step import crossval

def main():
    experiment = crossval.CrossVal('3OEC_current_flow.csv', 100, '5min',
                                   12, 'mps')
    save_dir = f"./out/crossval_100epochs_10runs_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    experiment.run_experiment(5, save_dir, early_stop=True)

if __name__=='__main__':
    main()