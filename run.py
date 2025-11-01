import argparse
from configs import get_cfg
from util.net import init_training
from util.util import run_pre, init_checkpoint
from trainer import get_trainer
import warnings
warnings.filterwarnings("ignore")

'''
python3 -m torch.distributed.launch --nproc_per_node=$nproc_per_node 
--nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr 
--master_port=$master_port --use_env run.py -c $1 -m $2
'''
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg_path', '-c', type=str, default='configs/dvad/dvad_mvtec.py', help='Path to the configuration file')
	parser.add_argument('--mode', '-m', type=str, default='train', choices=['train', 'test'], help='Mode of operation: train or test')
	parser.add_argument('--sleep', type=int, default=-1, help='Sleep duration (in seconds)')
	parser.add_argument('--memory', type=int, default=-1, help='Memory allocation limit')
	parser.add_argument('--dist_url', default='env://', type=str, help='URL used to set up distributed training')
	parser.add_argument('--logger_rank', default=0, type=int, help='GPU ID to use for logging')
	parser.add_argument('opts', nargs=argparse.REMAINDER, help='Additional options in key=value format')
	cfg_terminal = parser.parse_args()
 
	cfg = get_cfg(cfg_terminal)
	run_pre(cfg)
	init_training(cfg)
	init_checkpoint(cfg)
	trainer = get_trainer(cfg)
	trainer.run()


if __name__ == '__main__':
	main()
