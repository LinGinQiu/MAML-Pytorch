import  torch, os
import  numpy as np

from    no_blur_dataset import NoBlurTaskDataset
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from meta import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)
    maml = Meta(args, config).to(device)
    save_path= os.path.join(args.save_path, 'meta_learner_in_finnal.pth')
    maml.load_model(save_path)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    # --- Loading the model (for demonstration) ---

    print(maml)
    print('Total trainable tensors:', num)

    mini_val = NoBlurTaskDataset(mode='val',
                             batchsz=100, resize=args.imgsz)

    for epoch in range(args.epoch):
        # fetch meta_batchsz num of episode each time

        x_spt, y_spt, x_qry, y_qry = mini_val.create_task()

        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)

        print('Test acc:', accs)






if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=10)
    argparser.add_argument('--n_way', type=int, help='n way', default=4)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--batchsize', type=int, help='how many tasks used in a epoch', default=10000)


    argparser.add_argument('--data_path', type=str, help='data path', default='/scratch/cq2u24/Data/l2l/no_blur_dataset')
    argparser.add_argument('--save_path', type=str, help='save model path',
                           default='/scratch/cq2u24/Model/l2l/no_blur_dataset_checkpoints')

    arg = argparser.parse_args()
    # print(os.getcwd())
    if os.getcwd() == '/Users/qiuchuanhang/PycharmProjects/MAML-Pytorch':
        arg.data_path = '/Volumes/YuLin/no_blur_dataset'
        arg.save_path = '/Users/qiuchuanhang/PycharmProjects/MAML-Pytorch/backup'
        arg.epoch = 10
        arg.batchsize = 100

    main(args=arg)
