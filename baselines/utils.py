import os
import argparse
import logging
import random
import numpy as np


logger = logging.getLogger(__name__)



def cfg_parsing():
    parser = argparse.ArgumentParser()
    # Public parameters
    parser.add_argument("--mode", default=None, type=str, required=True)   
    parser.add_argument("--lang", default=None, type=str, required=True)   
    parser.add_argument("--output_path", default=None, type=str, required=True,
                        help="The path for output.")
    parser.add_argument("--seed", default=42, type=int,
                        help="The random seed.")
    
    # DPP parameters
    parser.add_argument("--num_candidates", default=16, type=int, 
                        help="The number of groups for training.")
    parser.add_argument("--num_icl", default=4, type=int, 
                        help="The number of examples in ICL.")  
    parser.add_argument("--num_process", default=8, type=int, 
                        help="The number of multi process.")  
    parser.add_argument("--dpp_topk", default=100, type=int,
                        help="The size of DPP matrix.")
    parser.add_argument("--scale_factor", default=0.1, type=float,
                        help="The factor to trade-off diversity and relevance.")
    parser.add_argument("--dimension", default=768, type=int,
                        help="The dimension of embedding.")
    
    # Inference parameters
    parser.add_argument("--model_name", default=None, type=str, required=True)
    parser.add_argument("--baseline_data_path", default='', type=str)
    parser.add_argument("--master_port", default='1', type=str)
    parser.add_argument("--generation_model_path", default=None, type=str) 
    parser.add_argument("--instruction", default='# optimize this code \n# slow version\n', type=str)
    parser.add_argument("--middle_instruction", default='\n# optimized version of the same code\n', type=str)
    parser.add_argument("--post_instruction", default='\n\n\n', type=str)
    parser.add_argument("--temperature", default=0, type=float)  
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--slice", default=1, type=int)
    parser.add_argument("--total", default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_gen_length", default=256, type=int)
    parser.add_argument("--max_seq_length", default=4096, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--testing_number", default=0, type=int)
    parser.add_argument("--do_train", action='store_true')
    

    # Training parameters
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--test_case_path", default=None, type=str)
    parser.add_argument("--process_number", default=30, type=int)
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--valid_batch_size", default=32, type=int)
    parser.add_argument("--margin", default=0.3, type=float)

    cfg = parser.parse_args()
    random.seed(cfg.seed)
    os.environ['PYHTONHASHSEED'] = str(cfg.seed)
    np.random.seed(cfg.seed)
    return cfg

