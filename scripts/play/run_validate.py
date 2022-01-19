import glob
import os
import subprocess

type_list = ['inst']

for path in glob.glob('data-bin/funcbound/data/*_*.bin', recursive=True):
    print(path)
    lib, opt = os.path.basename(path)[:-4].split('_')
    s = subprocess.run([r'python', 'validate.py', os.path.basename(path)[:-4],
                        '--mytype',
                        'func', '--path', 'checkpoints/funcbound/checkpoint_last.pt',
                        '--max-sentences', '8', '--user-dir', 'finetune_tasks', '--task',
                        'funcbound', '--required-batch-size-multiple', '1',
                        '--criterion', 'funcbound', '--num-classes', '2', '--write', '--myarch',
                        lib, '--opt', opt, '--cpu'])
