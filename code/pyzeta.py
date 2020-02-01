import os
from defaults import defaults
from itertools import product
import numpy as np


def get_patch_offsets(patch_radius, im_shape):
    """
    For given patch size, in a specific image, figure out the unraveled (flat index)
    offsets of each pixel in the patch relative to a central voxel.

    Works with 3D and 4D images.

    If the image is 4D, the first three dimensions are assumed to
    be spatial and the last dimension is assumed to correspond time/frames.

    If the image is 4D, the offsets are relative to the spatial centre
    of the patch in frame index 0


    :param patch_radius: Integer, patches are assumed to have odd
                         diameter where diam = 2 * radius.
    :param im_shape: Size of image array containing patch.


    :return: Flat index offsets of voxels in patch relative to central voxel.
    """

    im_dim = len(im_shape)

    assert im_dim in [3,4]

    # Spatial diameter.
    patch_diam = 1 + 2 * patch_radius

    # Can't exceed image spatial extents.
    for k in range(3):
        assert patch_diam <= im_shape[k]


    # Patch spatial extent
    ix_range = np.arange(patch_diam)
    ix_ranges = [ix_range] * 3

    # Spatial location of centre that we use to calculate the offsets.
    idx_cent = [patch_radius] * 3

    if im_dim == 4:
        # T-extent at the end.
        t_range = np.arange(im_shape[-1])
        ix_ranges.append(t_range)

        # Make the patch centre in the first frame of the 4D image
        idx_cent.append(0)


    flat_idx_cent = np.ravel_multi_index(idx_cent, im_shape)

    idx_offsets = []
    for multi_idx in product(*ix_ranges):
        flat_idx_curr = np.ravel_multi_index(multi_idx, im_shape)
        idx_offsets.append(flat_idx_curr - flat_idx_cent)

    return np.asarray(idx_offsets)


##############################################################

class PyZeta(object):

    def __init__(self, args):

        self.target = args.target
        self.ref_set = args.ref_set
        self.output = args.output

        self.mask = args.mask

        self.k_zeta = args.k_zeta
        self.patch_rad = args.patch_radius
        self.nbhd_rad = args.nbhd_radius

        self.dist = 'maha'
        if args.euclidean:
            self.dist = 'eucl'

        self.config_file = args.config_file

        if not (self.config_file is None):
            self.load_config_settings()

        self.report()


    ##############################################################


    def load_config_settings(self):
        print('Loading config settings from file:')
        print('      {:s}'.format(self.config_file))

        assert os.path.isfile(self.config_file)

        # TODO:
        # Stuff here ...


    ##############################################################


    def report(self):

        print('-------------------------------------------')
        print('Target: {:s}'.format(self.target))
        print('Refs:   {:s}'.format(self.ref_set))
        print('Output: {:s}'.format(self.output))

        if self.mask is None:
            print('No mask given.')
        else:
            print('Mask:   {:s}'.format(self.mask))

        print('k_zeta:    {:>2d}'.format(self.k_zeta))
        print('patch_rad: {:>2d}'.format(self.patch_rad))
        print('nbhd_rad:  {:>2d}'.format(self.nbhd_rad))
        print('dist:       {:s}'.format(self.dist))
        print('-------------------------------------------')


    ##############################################################


    def run(self):

        print('Running')


        if self.mask is None:
            # TODO: make a full mask
            self.mask = 1



    ##############################################################
