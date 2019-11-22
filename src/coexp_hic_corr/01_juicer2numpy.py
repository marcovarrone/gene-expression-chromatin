import argparse
import os

import pandas as pd
import scipy.sparse as sps

parser = argparse.ArgumentParser()

# ToDo: add description
parser.add_argument('-d', '--dataset', type=str, default='GM12878')
parser.add_argument('--type', type=str, choices=['observed', 'oe'], default='oe')
parser.add_argument('--norm', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR'], default='NONE')
parser.add_argument('--file', type=str, choices=['primary', 'replicate', 'combined'], default='combined')
parser.add_argument('--chr-src', type=int, default=11)
parser.add_argument('--chr-tgt', type=int, default=11)
parser.add_argument('--resolution', type=int, default=10000)
parser.add_argument('--force', default=False, action='store_true')
args = parser.parse_args()

data_folder = '/home/varrone/Data/{}/'.format(args.dataset)
hic_folder = '{}_{}_{}/'.format(args.file, args.type, args.norm)
output_file = '{}_{}_{}_{}'.format(args.file, args.chr_src, args.chr_tgt, args.resolution)

sps_path = data_folder + hic_folder + output_file
raw_path = sps_path + '.txt'

if not os.path.exists(data_folder + hic_folder):
    os.makedirs(data_folder + hic_folder)

if args.dataset == 'MCF7':
    hic_link = 'https://hicfiles.s3.amazonaws.com/external/barutcu/MCF-7.hic '
else:
    hic_link = 'https://hicfiles.s3.amazonaws.com/hiseq/{}/in-situ/{}.hic '.format(args.dataset.lower(), args.file)

if not os.path.exists(raw_path) or args.force:
    os.system('java -jar /home/varrone/Repo/software/juicer_tools_1.13.02.jar ' +
              'dump {} {} '.format(args.type, args.norm) +
              hic_link +
              '{} {} '.format(args.chr_src, args.chr_tgt) +
              'BP {} '.format(args.resolution) +
              '{}'.format(raw_path))

hic = pd.read_csv(raw_path, delim_whitespace=True, header=None)
contact_matrix = sps.csr_matrix(
    (hic.iloc[:, 2], (hic.iloc[:, 0] // args.resolution, hic.iloc[:, 1] // args.resolution)))
sps.save_npz(sps_path, contact_matrix)
os.remove(raw_path)
