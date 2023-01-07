import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import run_nerf
from run_nerf import config_parser
from vis import *
from utils import *
from run_nerf import render
import numpy as np
import matplotlib.pyplot as plt

dsid = 'flower'
parser = config_parser(f'configs/{dsid}.txt')
args = parser.parse_args()
args.i_weights = 1000
args.i_video = 50000
args.distill_active = True
args.render_only = False
args.factor = 8

args.dsid = dsid
args.expname = f'distill-nvd{int(args.no_viewdirs_distill)}-{dsid}'

# checkpoint to vanilla NeRF
args.basedir = './logs_official'
args.ft_path = f'logs_official/{dsid}_test/100000.tar'
if dsid == 'fern':
    args.ft_path = f'logs_official/{dsid}_test/200000.tar'
N_iters = int(args.ft_path.split('/')[-1].split('.')[0])
state = run_nerf.train(args, N_iters=N_iters + 5001)

# checkpoint to model trained with feature distillation
args.ft_path = f'logs_official/distill-nvd0-{dsid}/105000.tar'
if dsid == 'fern':
    args.ft_path = f'logs_official/distill-nvd0-{dsid}/205000.tar'
state = run_nerf.train(args, N_iters=0)

settings = load_settings()
img_i = settings[dsid]['view_a']
rgb, emb = render_composed(state, img_i)
r, c = settings[dsid]['rc']
extent = settings[dsid]['sz']
r = int(r * 8 / args.factor)
c = int(c * 8 / args.factor)
extent = int(extent * 8 / args.factor)

embq = calc_query_emb(emb, r, c, extent, rgb=rgb)
dist = calc_feature_dist(embq, emb)
emb_pca = calc_pca(emb)

plt.figure(figsize=(4,3))
plt.hist(dist.view(-1).cpu().numpy())
plt.title('Distances between query vector and retrieval vectors.')
plt.show()

img_j = settings[dsid]['view_b']
rgb_j, emb_j = render_composed(state, img_j)
rgb_j_fg, emb_j_fg = render_decomposed(
    state, img_j, embq, 
    dist_thr=settings[dsid]['thr'] + settings[dsid]['margin'], 
    foreground=True
)
rgb_j_bg, emb_j_bg = render_decomposed(
    state, img_j, embq, 
    dist_thr=settings[dsid]['thr'] - settings[dsid]['margin'], 
    foreground=False
)
f, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(rgb.cpu())
ax[0].axis('off')
ax[1].imshow(rgb_j.cpu())
ax[1].axis('off')
ax[2].imshow(rgb_j_fg.cpu())
ax[2].axis('off')
ax[3].imshow(rgb_j_bg.cpu())
ax[3].axis('off')
plt.show()

emb_dino = torch.from_numpy(state['features'][img_i])
embq_dino = calc_query_emb(emb_dino, r, c, extent).cpu()
dist_dino = calc_feature_dist(embq_dino, emb_dino)
emb_pca_dino = calc_pca(emb_dino)
f, ax = plt.subplots(2, 2)
# dino
ax[0,0].imshow(dist_dino, cmap='gray')
ax[0,0].axis('off')
ax[0,0].title.set_text('DINO')
ax[1,0].imshow(emb_pca_dino)
ax[1,0].axis('off')
# dino + nerf-n3f
ax[0,1].imshow(dist, cmap='gray')
ax[0,1].axis('off')
ax[0,1].title.set_text('DINO + NeRF-N3F')
ax[1,1].imshow(emb_pca)
ax[1,1].axis('off')
plt.show()