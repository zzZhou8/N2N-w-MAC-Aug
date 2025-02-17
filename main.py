import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from loader import get_loader
from solver import Solver

import torch
import random
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')

parser.add_argument('--save_epochs', type=int, default=5)
parser.add_argument('--test_epochs', type=int, default=110)
parser.add_argument('--decay_epochs', type=int, default=10)
parser.add_argument('--multi_gpu', type=bool, default=False)

parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--saved_path', type=str, default='./npy_img/')
parser.add_argument('--save_path', type=str, default='./save/')
parser.add_argument('--num_epochs', type=int, default=150)
parser.add_argument('--print_iters', type=int, default=20)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str)

args = parser.parse_args(args=[])

def main(args):
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    print('Preparing data')
    if args.mode == 'train':
        data_loader_train = get_loader(mode='train',
                                saved_path=args.saved_path,
                                batch_size=args.batch_size,
                                )
        
        data_loader = {'train':data_loader_train}
        
    else:
        data_loader_test = get_loader(mode=args.mode,
                                    saved_path=args.saved_path,
                                    batch_size=1,
                                    )
        
        data_loader = {'test':data_loader_test}

    print('The data is ready')
    solver = Solver(args, data_loader)
    print('Solver is ready')
    if args.mode == 'train':
        print('Lets start training.')
        solver.train()
    elif args.mode == 'test':
        print('Lets start testing')
        solver.test()

#Locked random number seed
def set_seed(seed):
    # 设置 Python 内置 random 模块的随机数种子
    random.seed(seed)
    
    # 设置环境变量 PYTHONHASHSEED，以确保哈希随机数生成器的种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 设置 NumPy 的随机数种子
    np.random.seed(seed)
    
    # 设置 PyTorch 的 CPU 随机数种子
    torch.manual_seed(seed)
    
    # 检查是否有可用的 GPU，再设置 CUDA 相关的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 所有 GPU 设同样的种子
        # 确保 CUDA 计算的可复现性（但可能影响性能）
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # 确保 PyTorch 使用确定性算法（适用于 PyTorch 1.8+）
    if torch.__version__ >= '1.8.0':
        torch.use_deterministic_algorithms(True)

    print(f"Random seed set to {seed}")


seed=random.randint(0,10000) #random set
#seed=1986 #pre-set
set_seed(seed)
main(args)

