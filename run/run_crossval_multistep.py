import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from experiments.multi_step import crossval

def main():
    experiment = crossval.CrossVal('3OEC_current_flow.csv', 400, '1min',
                                   90, 'cuda')
    save_dir = f"./out/crossval_100epochs_10runs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment.run_experiment(3, save_dir, early_stop=False)

if __name__=='__main__':
    main()