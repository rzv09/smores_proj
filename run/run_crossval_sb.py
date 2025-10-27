import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from experiments.single_step.crossval_sb import CrossValSB

def main():
    experiment = CrossValSB('DO_allsites_allyears_20250611.csv', 100, '5min',
                                   12, 'cuda')
    save_dir = f"./out/crossval_100epochs_10runs_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    experiment.run_experiment(5, save_dir, early_stop=True)

if __name__=='__main__':
    main()