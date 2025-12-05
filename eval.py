import os
from models.build import create_model
import torch
from dataclasses import dataclass
from models.tokenizer import Tokenizer
from torch.utils.data import DataLoader
from util.datasets import All, dual_collate_fn, single_collate_fn
from util.misc import setup_for_distributed
from util.misc import MetricLogger, Evaluator
import logging
from datetime import datetime
dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelArgs_7B:
    llama_model_path = '/ai/teacher/dkc/Assets/origin/weights/'
    llm_model = '7B'
    max_seq_len = 512
    hidden_proj = 128
    emb = 69
    adapter_scale = 0.1
    adapter_dim = 12
    cpu_load = False
    gradient_checkpointing = False
    is_train = False
    compression_level=3 


def main(args):
    args.mem_type = ('dual' if args.model_name in ['0_ours', '1_zipped', '2_full', '3_hierachical', '3_rebuttal_abstract', '3_rebuttal_compress'] else 'single')
    dataset = All(args.dataset, 'val', args.mem_type)
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=dual_collate_fn if args.mem_type == 'dual' else single_collate_fn)
    evaluator = Evaluator()
    
    llama = create_model(args)
    adapter = torch.load(f'./ckpts/{args.dataset}-{args.model_name}-ckpt-19.pth', weights_only=False)['model']
    sd = {}
    for k in adapter:
        print(k)
        sd[k.replace('module.', '')] = adapter[k]

    llama.load_state_dict(sd, False)

    tokenizer = Tokenizer(model_path=os.path.join(args.llama_model_path, 'tokenizer.model'))
    metric_logger = MetricLogger(delimiter=" ")
    
    count = 0
    correct = 0
    quit_flag = False
    for item in metric_logger.log_every(iterable=dataloader, print_freq=1):
        if args.mem_type == 'dual':
            qs, anss, imgs, hf_imgs, indicators = item
            preds = llama.generate(qs, imgs, hf_imgs, indicators, 20, tokenizer)
        else:
            qs, anss, imgs, indicators = item
            preds = llama.generate(qs, imgs, indicators, 20, tokenizer)
        for idx, pred in enumerate(preds):
            count += 1
            if evaluator.evaluate(pred, anss[idx]):
                correct += 1
                print(f'{qs[idx]:<30} pred: {pred:<20} gt: {str(anss[idx]):<20} correct {correct * 100 /count:.2f}% {correct}/{count}\n')
            else:
                print(f'{qs[idx]:<30} pred: {pred:<20} gt: {str(anss[idx]):<20} wrong   {correct * 100 /count:.2f}% {correct}/{count}\n')
        
        
        
        

    print(f'Overall: {correct * 100 /count:.2f}% {correct}/{count}')

def get_arg_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str) 
    parser.add_argument('--model_name', type=str)
    return parser


if __name__ == '__main__':
    setup_for_distributed(True) 
    cmd_args = get_arg_parser().parse_args() 

    args = ModelArgs_7B()
    args.dataset = cmd_args.dataset
    args.model_name = cmd_args.model_name
    log_root = Path(f'./rebuttal_val_logs/')
    log_root.mkdir(exist_ok=True)
    log_file_name = log_root / f'{dt}-Val-{args.dataset}-{args.model_name}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode='w',
        filename=log_file_name.__str__()
    )
    main(args)