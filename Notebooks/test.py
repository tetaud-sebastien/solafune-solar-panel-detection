import glob
import tifffile
import numpy as np
import lightgbm as lgb
import warnings
import tqdm

warnings.simplefilter('ignore')

from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier


train_path =  '/home/sebastien/Documents/projects/solafune-solar-panel-detection/data/train/s2_image/'
mask_path = '/home/sebastien/Documents/projects/solafune-solar-panel-detection/data/train/mask/'

masks = glob.glob(f'{mask_path}/*')
trains = glob.glob(f'{train_path}/*')
masks.sort()
trains.sort()

X = []
y = []
g = []

for i, (t, m) in enumerate(zip(trains, masks)):
    img = tifffile.imread(t).astype(np.float64)
    mask = tifffile.imread(m).astype(np.float64)
    X.append(img.reshape(-1,12))
    y.append(mask.reshape(-1))
    g.append(np.ones_like(mask.reshape(-1))*i)
    
X = np.vstack(X)
y = np.hstack(y)
g = np.hstack(g)




X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=100, population_size=100,
                          offspring_size=None, mutation_rate=0.9,
                          crossover_rate=0.1,
                          scoring='f1', cv=5,
                          subsample=1.0, n_jobs=14,
                          max_time_mins=10, max_eval_time_mins=10,
                          random_state=None, config_dict=None,
                          template=None,
                          warm_start=False,
                          memory=None,
                          use_dask=False,
                          periodic_checkpoint_folder=True,
                          early_stop=True,
                          verbosity=3,
                          disable_update_check=False,
                          log_file=None
                          )
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_pipeline.py')