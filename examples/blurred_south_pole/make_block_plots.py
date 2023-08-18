import itertools as it
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import flux.compressed_form_factors as cff
import flux.compressed_form_factors_compare as cffc
from flux.compressed_form_factors import CompressedFormFactorMatrix

import scipy
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--compression_type', type=str, default="svd", choices=["svd", "rand_svd", "aca", "rand_id", "compare"])
parser.add_argument('--max_area', type=float, default=3.0)
parser.add_argument('--outer_radius', type=int, default=80)
parser.add_argument('--sigma', type=float, default=5.0)

parser.add_argument('--tol', type=float, default=1e-1)
parser.add_argument('--min_depth', type=int, default=1)
parser.add_argument('--max_depth', type=int, default=0)
parser.add_argument('--compress_sparse', action='store_true')

parser.add_argument('--add_residuals', action='store_true')
parser.add_argument('--k0', type=int, default=40)
parser.add_argument('--p', type=int, default=5)
parser.add_argument('--q', type=int, default=1)

parser.add_argument('--cliques', action='store_true')
parser.add_argument('--n_cliques', type=int, default=25)
parser.add_argument('--obb', action='store_true')

parser.add_argument('--plot_labels', action='store_true')

parser.set_defaults(feature=False)

args = parser.parse_args()


def plot_blocks(block, fig, **kwargs):

    if 'figsize' in kwargs:
        fig.set_size_inches(*kwargs['figsize'])
    else:
        fig.set_size_inches(12, 12)

    ax = fig.add_subplot()
    ax.axis('off')

    ax.set_xlim(-0.001, 1.001)
    ax.set_ylim(-0.001, 1.001)

    rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='k',
                             facecolor='none')
    ax.add_patch(rect)

    def get_ind_offsets(inds):
        return np.concatenate([[0], np.cumsum([len(I) for I in inds])])

    def add_rects(block, c0=(0, 0), w0=1, h0=1):
        row_offsets = get_ind_offsets(block._row_block_inds)
        col_offsets = get_ind_offsets(block._col_block_inds)
        for (i, row), (j, col) in it.product(
            enumerate(row_offsets[:-1]),
            enumerate(col_offsets[:-1])
        ):
            w = w0*(row_offsets[i + 1] - row)/row_offsets[-1]
            h = h0*(col_offsets[j + 1] - col)/col_offsets[-1]
            i0, j0 = c0
            c = (i0 + w0*row/row_offsets[-1], j0 + h0*col/col_offsets[-1])

            child = block._blocks[i, j]
            if child.is_leaf:
                if isinstance(child, cff.FormFactorSvdBlock) or isinstance(child, cff.FormFactorSparseSvdBlock) or \
                isinstance(child, cff.FormFactorAcaBlock) or isinstance(child, cff.FormFactorSparseAcaBlock) or \
                isinstance(child, cff.FormFactorIdBlock) or isinstance(child, cff.FormFactorSparseIdBlock):
                    facecolor = 'cyan' if child.compressed else 'orange'
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor=facecolor)
                    ax.add_patch(rect)
                elif isinstance(child, cff.FormFactorZeroBlock):
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor='black')
                    ax.add_patch(rect)
                elif isinstance(child, cff.FormFactorSparseBlock):
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor='white')
                    ax.add_patch(rect)
                elif isinstance(child, cff.FormFactorDenseBlock):
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor='magenta')
                    ax.add_patch(rect)
                elif isinstance(child, cff.FormFactorNullBlock):
                    continue
                else:
                    raise Exception('TODO: add %s to _plot_block' % type(child))

                if args.plot_labels:
                    ax.text(c[0] + 0.5*w, c[1] + 0.5*h, "{:.0e}".format(np.prod(child.shape)), transform=ax.transAxes, fontsize=8, verticalalignment='center', horizontalalignment='center')
            else:
                add_rects(child, c, w, h)

            rect = patches.Rectangle(
                c, w, h, linewidth=1, edgecolor='k', facecolor='none')
            ax.add_patch(rect)

    add_rects(block)

    ax.invert_xaxis()

    return fig, ax


def plot_blocks_compare(block, fig, **kwargs):

    if 'figsize' in kwargs:
        fig.set_size_inches(*kwargs['figsize'])
    else:
        fig.set_size_inches(12, 12)

    ax = fig.add_subplot()
    ax.axis('off')

    ax.set_xlim(-0.001, 1.001)
    ax.set_ylim(-0.001, 1.001)

    rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='k',
                             facecolor='none')
    ax.add_patch(rect)

    def get_ind_offsets(inds):
        return np.concatenate([[0], np.cumsum([len(I) for I in inds])])

    def add_rects(block, c0=(0, 0), w0=1, h0=1):
        row_offsets = get_ind_offsets(block._row_block_inds)
        col_offsets = get_ind_offsets(block._col_block_inds)
        for (i, row), (j, col) in it.product(
            enumerate(row_offsets[:-1]),
            enumerate(col_offsets[:-1])
        ):
            w = w0*(row_offsets[i + 1] - row)/row_offsets[-1]
            h = h0*(col_offsets[j + 1] - col)/col_offsets[-1]
            i0, j0 = c0
            c = (i0 + w0*row/row_offsets[-1], j0 + h0*col/col_offsets[-1])

            child = block._blocks[i, j]
            if child.is_leaf:
                if isinstance(child, cff.FormFactorSvdBlock):
                    facecolor = '#1f77b4'
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor=facecolor)
                    ax.add_patch(rect)

                elif isinstance(child, cff.FormFactorSparseSvdBlock):
                    facecolor = '#ff7f0e'
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor=facecolor)
                    ax.add_patch(rect)

                elif isinstance(child, cff.FormFactorAcaBlock):
                    facecolor = '#FF3936'
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor=facecolor)
                    ax.add_patch(rect)

                elif isinstance(child, cff.FormFactorSparseAcaBlock):
                    facecolor = '#7f7f7f'
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor=facecolor)
                    ax.add_patch(rect)

                elif isinstance(child, cff.FormFactorIdBlock):
                    facecolor = '#0CFF00'
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor=facecolor)
                    ax.add_patch(rect)

                elif isinstance(child, cff.FormFactorSparseIdBlock):
                    facecolor = '#17becf'
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor=facecolor)
                    ax.add_patch(rect)

                elif isinstance(child, cff.FormFactorZeroBlock):
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor='black')
                    ax.add_patch(rect)
                elif isinstance(child, cff.FormFactorSparseBlock):
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor='white')
                    ax.add_patch(rect)
                elif isinstance(child, cff.FormFactorDenseBlock):
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor='magenta')
                    ax.add_patch(rect)
                elif isinstance(child, cff.FormFactorNullBlock):
                    continue
                else:
                    raise Exception('TODO: add %s to _plot_block' % type(child))

                if args.plot_labels:
                    ax.text(c[0] + 0.5*w, c[1] + 0.5*h, "{:.0e}".format(np.prod(child.shape)), transform=ax.transAxes, fontsize=8, verticalalignment='center', horizontalalignment='center')
            else:
                add_rects(child, c, w, h)

            rect = patches.Rectangle(
                c, w, h, linewidth=1, edgecolor='k', facecolor='none')
            ax.add_patch(rect)

    add_rects(block)

    ax.invert_xaxis()

    return fig, ax




compression_type = args.compression_type
max_area_str = str(args.max_area)
outer_radius_str = str(args.outer_radius)
sigma_str = str(args.sigma)

max_depth = args.max_depth if args.max_depth != 0 else None

if compression_type == "svd" or compression_type == "aca":
    if args.add_residuals:
        FF_dir = "{}_resid_{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, sigma_str, args.tol,
        args.k0)
    else:
        FF_dir = "{}_{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, sigma_str, args.tol,
            args.k0)

elif compression_type == "rand_svd" or compression_type == "rand_id" or compression_type == "compare":
    if args.add_residuals and not compression_type == "compare":
        FF_dir = "{}_resid_{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, sigma_str, args.tol,
            args.p, args.q, args.k0)
    else:
        FF_dir = "{}_{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, sigma_str, args.tol,
            args.p, args.q, args.k0)


if not args.cliques and not compression_type == "compare" and args.min_depth != 1:
    FF_dir += "_{}mindepth".format(args.min_depth)

if max_depth is not None:
    FF_dir += "_{}maxdepth".format(max_depth)

if args.compress_sparse:
    FF_dir += "_cs"

result_dir = "results"
if args.cliques:
    FF_dir += "_{}nc".format(args.n_cliques)
    result_dir += "_cliques"

if args.cliques and args.obb:
    FF_dir = FF_dir + "_obb"

if args.add_residuals and not compression_type == "compare":
    FF_path =  result_dir+"/"+ FF_dir + "/FF_{}_{}_{:.0e}_{}_resid.bin".format(max_area_str, outer_radius_str, args.tol, compression_type)
else:
    FF_path =  result_dir+"/"+ FF_dir + "/FF_{}_{}_{:.0e}_{}.bin".format(max_area_str, outer_radius_str, args.tol, compression_type)

if not os.path.exists(FF_path):
    print("PATH DOES NOT EXIST " + FF_path)
    assert False
FF = CompressedFormFactorMatrix.from_file(FF_path)


fig = plt.figure(figsize=(18, 6))  # , figsize=(18, 6))
print(f'- {FF_path}')
if compression_type == "compare":
    fig, ax = plot_blocks_compare(FF._root, fig)
    
    legend_elements = [
        patches.Patch(facecolor='#1f77b4',label='SVD'),
        patches.Patch(facecolor='#ff7f0e',label='SVD + Resid'),
        patches.Patch(facecolor='#FF3936',label='ACA'),
        patches.Patch(facecolor='#7f7f7f',label='ACA + Resid'),
        patches.Patch(facecolor='#0CFF00',label='Random ID'),
        patches.Patch(facecolor='#17becf',label='Random ID + Resid'),
    ]
    
    ax.legend(handles=legend_elements)
else:
    fig, ax = plot_blocks(FF._root, fig)
if args.plot_labels:
    fig.savefig(result_dir+"/"+FF_dir+"/block_plot_labeled.png")
else:
    fig.savefig(result_dir+"/"+FF_dir+"/block_plot.png")
plt.close(fig)