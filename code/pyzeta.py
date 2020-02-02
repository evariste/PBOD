import os
from defaults import defaults
from itertools import product
import numpy as np
import nibabel as nib

##############################################################

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

        self.target_name = args.target
        self.ref_set_name = args.ref_set
        self.output = args.output

        self.mask_name = args.mask

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

        # Stuff that will be set up later
        self.mask = None
        self.tgt = None
        self.ref = None

        self.tgt_dat = None
        self.ref_dat = None


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
        print('Target: {:s}'.format(self.target_name))
        print('Refs:   {:s}'.format(self.ref_set_name))
        print('Output: {:s}'.format(self.output))

        if self.mask_name is None:
            print('No mask given.')
        else:
            print('Mask:   {:s}'.format(self.mask_name))

        print('k_zeta:    {:>2d}'.format(self.k_zeta))
        print('patch_rad: {:>2d}'.format(self.patch_rad))
        print('nbhd_rad:  {:>2d}'.format(self.nbhd_rad))
        print('dist:       {:s}'.format(self.dist))
        print('-------------------------------------------')


    ##############################################################


    def initialise(self):

        print('Initialise')

        assert os.path.isfile(self.target_name)
        assert os.path.exists(self.ref_set_name)

        if not (self.mask_name is None):
            assert os.path.isfile(self.mask_name)


        self.tgt = nib.load(self.target_name)

        if os.path.isfile(self.ref_set_name):
            self.ref = nib.load(self.ref_set_name)
        else:
            # TODO: Deal with directory of ref set data.
            raise Exception('Ref set as a directory: TODO')


        if self.mask_name is None:
            # TODO: make a full mask
            self.mask = None
        else:
            self.mask = nib.load(self.mask_name)



    ##############################################################

    def run(self):

        print('Running')

        self.tgt_dat = self.tgt.get_fdata()
        self.ref_dat = self.ref.get_fdata()

        if self.mask is None:
            self.gen_default_mask()
        else:
            self.mask_dat = self.mask.get_fdata()


        self.check_input_data()


    ##############################################################


    def gen_default_mask(self):

        sz = self.tgt_dat.shape
        self.mask = np.ones(sz, dtype=np.uint8)
        return

    ##############################################################

    def check_input_data(self):

        sz_mask = self.mask_dat.shape
        sz_tgt = self.tgt_dat.shape
        sz_ref = self.ref_dat.shape

        # Dimensions
        assert len(sz_mask) == 3
        assert len(sz_tgt) in [3,4]
        assert len(sz_ref) in [3,4,5]

        # Check compatibility - spatial dimensions
        assert sz_mask == sz_tgt[:3]
        assert sz_mask == sz_ref[:3]

        # Target and ref compatible
        if len(sz_tgt) == 3:
            assert len(sz_ref) in [3,4]

        if len(sz_tgt) == 4:
            assert len(sz_ref) in [4,5]
            assert sz_tgt == sz_ref[:4]


        # Zero out mask where not feasible
        margin = self.nbhd_rad + self.patch_rad

        ix = np.ix_(range(0,margin))
        modified = False
        if np.any(self.mask_dat[ix,:,:]):
            self.mask_dat[ix,:,:] = 0
            modified = True
        if np.any(self.mask_dat[:,ix,:]):
            self.mask_dat[:,ix,:] = 0
            modified = True
        if np.any(self.mask_dat[:,:,ix]):
            self.mask_dat[:,:,ix] = 0
            modified = True
        ix = np.ix_(-margin,0)
        if np.any(self.mask_dat[ix,:,:]):
            self.mask_dat[ix,:,:] = 0
            modified = True
        if np.any(self.mask_dat[:,ix,:]):
            self.mask_dat[:,ix,:] = 0
            modified = True
        if np.any(self.mask_dat[:,:,ix]):
            self.mask_dat[:,:,ix] = 0
            modified = True

        if modified and (not (self.mask_name is None)):
            print('Warning needed to set foreground mask voxels to zero.')
            print('Too close to volume edge.')


    ##############################################################
