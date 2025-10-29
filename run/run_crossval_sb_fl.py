import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from experiments.single_step.crossval_sb_fl import CrossValSBFL

def main():
    runs = 1
    experiment = CrossValSBFL('DO_allsites_allyears_20250611.csv', '3OEC_current_flow.csv', 100, '5min',
                                   12, 'mps')
    save_dir = f"./out/crossval_SB_FL_100epochs_{runs}runs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment.run_experiment(runs, save_dir, early_stop=True)

if __name__=='__main__':
    main()