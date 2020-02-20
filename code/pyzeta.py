import os
from defaults import defaults
from itertools import product
import numpy as np
import nibabel as nib

##############################################################

def get_patch_offsets(patch_rad, sz_img, strides, data_item_size):
    """
        For given patch size, in an image array with the given size and
        strides, figure out the unraveled (flat index) offsets of each
        voxel in a patch relative to its central voxel.

        NB works for 3D and 4D images only.

        If the image is 4D, the first three dimensions are assumed to
        be spatial and the last dimension is assumed to correspond
        time/frames.

        If the image is 4D, the offsets are relative to the spatial
        centre of the patch in frame index 0


        :param patch_rad: Integer, patches are assumed to have odd
               diameter where diam = 2 * radius + 1.
        :param sz_img: Size of image array.
        :param strides: Strides in bytes of elements in the array.
        :param data_item_size: The number of bytes for each element,
               assumed to be uniform.


        :return: Flat index offsets of voxels in patch relative to
                 central voxel.
        """

    assert len(strides) in [3,4]

    s = np.asarray(strides)

    assert np.all(s % data_item_size == 0)

    idx_strides = s // data_item_size

    f_order = np.all(idx_strides[:-1] < idx_strides[1:])
    c_order = np.all(idx_strides[:-1] > idx_strides[1:])

    assert (f_order or c_order)

    # spatial patch indices
    rx = range(-patch_rad, patch_rad+1)
    rxs = 3 * [rx]

    # 'C' order by default, last index changes fastest.
    ix = list(product(*rxs))

    if f_order:
        # Reverse entries so that we get Fortran order.
        # first index changes fastest.
        ix = [x[::-1] for x in ix]

    offset_i, offset_j, offset_k = idx_strides[:3]

    offset_t = 0
    t_dim = 1

    if len(strides) == 4:
        offset_t = idx_strides[-1]
        assert len(sz_img) == 4
        t_dim = sz_img[-1]


    vol_patch = t_dim * (1+2*patch_rad)**3

    offsets = np.zeros((vol_patch,), dtype=np.int64)


    n = 0
    for t in range(t_dim):
        for i,j,k in ix:
            offsets[n] = i * offset_i + j * offset_j + k * offset_k + t * offset_t
            n += 1

    return np.atleast_2d(offsets)

##############################################################

def get_data_order(strides, strict_check=True):

    s = np.asarray(strides)
    f_order = np.all(s[:-1] < s[1:])
    c_order = np.all(s[:-1] > s[1:])

    if strict_check:
        assert (f_order or c_order)

    if c_order:
        order = 'C'
    elif f_order:
        order = 'F'
    else:
        order = 'unknown'

    return order


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
        self.ref_order = 'F'

        self.tgt_dat = None
        self.ref_dat = None

        return

    ##############################################################


    def load_config_settings(self):
        print('Loading config settings from file:')
        print('      {:s}'.format(self.config_file))

        assert os.path.isfile(self.config_file)

        # TODO:
        # Stuff here ...

        return

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

        return


    ##############################################################

    def load_refs(self):

        # Reference set
        if os.path.isfile(self.ref_set_name):
            self.load_refs_from_file()
        elif os.path.isdir(self.ref_set_name):
            self.load_refs_from_dir()
        else:
            raise Exception('Cannot load reference data.')

    ##############################################################

    def load_refs_from_file(self):

        self.ref = nib.load(self.ref_set_name)

        return

    ##############################################################

    def load_refs_from_dir(self):
        print('Loading reference data from directory')
        d = self.ref_set_name
        fs = os.listdir(d)

        imgs_dat = []

        for f in fs:

            try:
                ff = os.path.join(d, f)
                img = nib.load(ff)
            except:
                print('Cannot load file: {:s}')
                continue
            imgs_dat.append(img.get_fdata())

        assert len(imgs_dat) > 0

        sz = list(set([x.shape for x in imgs_dat]))
        assert len(sz) == 1 , 'Ref. data: size mismatch'

        strides = list(set([x.strides for x in imgs_dat]))
        assert len(strides) == 1 , 'Ref. data: strides mismatch'

        dt = list(set([x.dtype for x in imgs_dat]))
        assert len(dt) == 1 , 'Ref. data: type mismatch'

        sz, strides, dt = sz[0], strides[0], dt[0]


        n_imgs = len(imgs_dat)

        self.ref_order = get_data_order(strides)

        dat_all = np.zeros(sz+(n_imgs,), dtype=dt, order=self.ref_order)

        for n in range(n_imgs):
            dat_all[...,n] = imgs_dat[n]

        self.ref = nib.Nifti1Image(dat_all, np.eye(4))

        return

    ##############################################################

    def initialise(self):

        print('Initialise')

        assert os.path.isfile(self.target_name)
        assert os.path.exists(self.ref_set_name)

        if not (self.mask_name is None):
            assert os.path.isfile(self.mask_name)


        # Reference set:
        self.load_refs()

        # Target:
        self.tgt = nib.load(self.target_name)

        # Mask:
        if self.mask_name is None:
            # TODO: make a full mask
            self.mask = None
        else:
            self.mask = nib.load(self.mask_name)



        # Data:
        self.ref_dat = self.ref.get_fdata()

        self.tgt_dat = self.tgt.get_fdata()

        if self.mask is None:
            self.gen_default_mask()

        self.mask_dat = self.mask.get_fdata()


        return

    ##############################################################

    def run(self):

        self.initialise()

        print('Running')



        self.check_input_data()




        print('Done.')



        return

    ##############################################################


    def gen_default_mask(self):

        self.mask = np.ones_like(self.tgt_dat, dtype=np.uint8, order='K')
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

        ix = range(0,margin)
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

        ix = range(-margin,0)
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

        return

    ##############################################################
