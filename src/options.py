import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_users', type=int, default=20)
    parser.add_argument('--z_std', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--local_ep', type=int, default=5)
    parser.add_argument('--local_bs', type=int, default=100)
    parser.add_argument('--sample_size', type=int, default=1000)
    
    args = parser.parse_args()
    return args