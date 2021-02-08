import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='MyModel')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--word_vec_dir', default='glove/glove.6B.50d.txt')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--processed_data_dir', default='_processed_data')
    parser.add_argument('--encoder', default='PCNN', type=str, help='PCNN, CNN or BiGRU')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--max_length', default=120, type=int)
    parser.add_argument('--max_pos_length', default=100, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr', default=0.5, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--seed', default=2021, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--early_stop', default=20, type=int)
    return vars(parser.parse_args())
