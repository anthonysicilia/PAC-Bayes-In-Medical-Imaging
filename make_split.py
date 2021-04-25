import random

from pathlib import Path
import os
from PIL import Image

TASK1_IMG_DIR = 'ISIC2018/ISIC2018_Task1-2_Training_Input'
TASK3_IMG_DIR = 'ISIC2018/ISIC2018_Task3_Training_Input'

def isimage(fname):

    try:
        _ = Image.open(fname)
        return True
    except IOError:
        return False
    
    raise NotImplementedError('Unhandled case encountered.')

def write(paths, fname):
    with open(fname, 'w') as out:
        for p in paths:
            out.write(f'{p}\n')

if __name__ == '__main__':

    random.seed(0)

    tasks = [(TASK1_IMG_DIR, 'task1'),
        (TASK3_IMG_DIR, 'task3')]

    for task_dir, task_name in tasks:

        path = os.path.join('path', task_name)

        Path(path).mkdir(exist_ok=True, parents=True)
        task_files = [fname for fname in os.listdir(task_dir)
            if isimage(os.path.join(task_dir,fname))]
        
        train = []
        final_holdout = []
        
        for fname in task_files:
            if random.random() <= 0.9:
                train.append(fname)
            else:
                final_holdout.append(fname)
        
        write(final_holdout, 
            os.path.join(path, 'final_holdout.txt'))
        write(train, 
            os.path.join(path, 'pac_bayes_full_train.txt'))

        hoeffding_holdout = []
        hoeffding_train = []

        for fname in train:
            if random.random() <= 0.9:
                hoeffding_train.append(fname)
            else:
                hoeffding_holdout.append(fname)
        
        write(hoeffding_holdout, 
            os.path.join(path, 'hoeffding_holdout.txt'))
        write(hoeffding_train, 
            os.path.join(path, 'hoeffding_train.txt'))
        
        prefix = []
        bound = []

        for fname in train:
            if random.random() <= 0.5:
                bound.append(fname)
            else:
                prefix.append(fname)
        
        write(prefix, 
            os.path.join(path, 'pac_bayes_prefix.txt'))
        write(bound, 
            os.path.join(path, 'pac_bayes_prefix_bound.txt'))

        

        








