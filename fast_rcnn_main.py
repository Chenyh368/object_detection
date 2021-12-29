from utils.experiman import manager
from utils.distributed_utils import init_distributed_mode
import os
from torch.multiprocessing import Process
import torch
import torch.distributed as dist

def add_parser_argument(parser):
    ## ======================== Data ==========================
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--num_classes', type=int, default=5)
    ## ======================== Training ==========================
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    ## ==================== Optimization ======================
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default ='sgd')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    ## ==================== Multi GPU ======================
    parser.add_argument('--world_size', default=3, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

def main(rank, world_size, opt, manager):
    manager.set_rank(rank)
    # 初始化各进程环境 start
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    opt.rank = rank
    opt.world_size = world_size
    opt.gpu = rank
    opt.distributed = True

    torch.cuda.set_device(opt.gpu)
    opt.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        opt.rank, opt.dist_url), flush=True)
    torch.distributed.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                         world_size=opt.world_size, rank=opt.rank)
    torch.distributed.barrier()
    if manager.is_master():
        logger.info(opt)



if __name__ == "__main__":
    parser = manager.get_basic_arg_parser()
    add_parser_argument(parser)
    opt = parser.parse_args()

    manager.setup(opt, third_party_tools=('tensorboard',))
    logger = manager.get_logger()
    device = 'cuda'

    logger.info('==> Preparing for Multi GPU')
    world_size = opt.world_size
    processes = []
    for rank in range(world_size):
        p = Process(target=main, args=(rank, world_size, opt, manager))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()