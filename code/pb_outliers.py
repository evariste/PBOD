import os
from defaults import defaults
from itertools import product
import numpy as np
import nibabel as nib
from sklearn.metrics.pairwise import euclidean_distances



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

class PBOutliers(object):

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
        self.ref = None
        self.ref_order = 'F'
        self.ref_dat = None

        self.tgt = None
        self.tgt_dat = None

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

        self.ref_dat = self.ref.get_fdata()

        return

    ##############################################################

    def load_refs_from_file(self):

        self.ref = nib.load(self.ref_set_name, mmap=False)

        self.ref_order = self.ref.dataobj.order

        assert self.ref_order in ['F', 'C']

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


        del imgs_dat

        self.ref = nib.Nifti1Image(dat_all, np.eye(4))

        return

    ##############################################################

    def load_target(self):

        self.tgt = nib.load(self.target_name, mmap=False)


        self.tgt_dat = self.tgt.get_fdata() # type: np.ndarray


        tgt_order = get_data_order(self.tgt_dat.strides)

        if tgt_order == self.ref_order:
            return

        # Need to re-order data of target to match that of reference.

        aff = self.tgt.get_affine()
        dt = self.tgt_dat.dtype
        sz = self.tgt_dat.shape
        dim = len(sz)

        assert dim in [3,4]


        if not(dim == 3):
            # TODO:
            raise Exception('Only 3-D target images implemented. 4-D not yet done.')


        tmp = np.zeros(sz, dtype=dt, order=self.ref_order)

        rxs = [range(sz[k]) for k in range(dim)]
        ix = list(product(*rxs))
        for i, j, k in ix:
            tmp[i,j,k] = self.tgt_dat[i,j,k]


        nii = nib.Nifti1Image(tmp, aff, header=self.tgt.header)

        del self.tgt_dat
        del self.tgt

        self.tgt = nii
        self.tgt_dat = tmp

        return

    ##############################################################

    def load_mask(self):

        if self.mask_name is None:
            # 'Full' mask with zeros at edges.
            self.mask_dat = np.ones(self.tgt_dat, dtype=np.uint8, order=self.ref_order)

            margin = self.patch_rad + self.nbhd_rad

            ix = range(0, margin)
            self.mask_dat[ix, :, :] = 0
            self.mask_dat[:, ix, :] = 0
            self.mask_dat[:, :, ix] = 0

            ix = range(-margin, 0)
            self.mask_dat[ix, :, :] = 0
            self.mask_dat[:, ix, :] = 0
            self.mask_dat[:, :, ix] = 0

        else:
            nii_mask = nib.load(self.mask_name, mmap=False)
            self.mask_dat = nii_mask.get_fdata()


        mask_order = get_data_order(self.mask_dat.strides)

        if mask_order == self.ref_order:
            return


        # Need to re-order data of mask to match that of reference.

        dt = np.uint8
        sz = self.mask_dat.shape
        dim = len(sz)

        assert dim == 3


        tmp = np.zeros(sz, dtype=dt, order=self.ref_order)

        rxs = [range(sz[k]) for k in range(dim)]
        ix = list(product(*rxs))
        for i, j, k in ix:
            if self.mask_dat[i,j,k] > 0:
                tmp[i,j,k] = 1


        del self.mask_dat

        self.mask_dat = tmp


        return

    ##############################################################

    def initialise(self):

        print('Initialise')

        assert os.path.isfile(self.target_name)
        assert os.path.exists(self.ref_set_name)

        if not (self.mask_name is None):
            assert os.path.isfile(self.mask_name)


        # First load reference set.
        self.load_refs()

        self.load_target()

        self.load_mask()



        return

    ##############################################################

    def run(self):

        self.initialise()


        self.check_input_data()

        print('Running')


        mask_flat = self.mask_dat.ravel(order='A')

        refs_flat = self.ref_dat.ravel(order='A')

        tgt_flat = self.tgt_dat.ravel(order='A')

        tgt_inds = np.argwhere(mask_flat > 0).ravel()

        out_dat = np.zeros_like(self.tgt_dat, order='A')
        out_dat_flat = out_dat.ravel(order='A')


        # Offsets for the centres of reference patches in a neighbourhood,
        # relative to the central voxel of interest.
        nbhd_offsets = get_patch_offsets(self.nbhd_rad,
                                         self.ref_dat.shape,
                                         self.ref_dat.strides,
                                         self.ref_dat.itemsize)
        # Make them a column vector.
        nbhd_offsets = nbhd_offsets.T


        # Offsets of patch voxels relative to central voxel.
        patch_offsets = get_patch_offsets(self.patch_rad,
                                          self.tgt_dat.shape,
                                          self.tgt_dat.strides,
                                          self.tgt_dat.itemsize)

        # How big is the neighbourhood?
        n_nbhd = nbhd_offsets.size
        # Patch volume.
        patch_vol = patch_offsets.size

        # Prepare arrays for storing stuff inside the main loop. Try to
        # avoid overhead of claiming memory within the loop.

        # Indices of all patch voxels in the neigbourhood.
        patch_inds = np.zeros((n_nbhd, patch_vol), dtype=np.int64)

        # Flat view of above.
        patch_inds_ravel = np.ravel(patch_inds, order='A')


        # Storage for contents of all ref patches:
        patch_dat_refs = np.zeros((n_nbhd, patch_vol), dtype=np.float)
        # Flat view of above.
        patch_dat_refs_ravel = patch_dat_refs.ravel(order='A')


        # Patch data for the nearest k reference patches to the target.
        patch_dat_refs_knn = np.zeros((self.k_zeta, patch_vol), dtype=np.float)
        patch_dat_refs_knn_ravel = np.ravel(patch_dat_refs_knn, order='A')

        # Indices in the upper half of a distance matrix (i.e. ignoring zeros in
        # diagonal and lower half which is same as upper half for symmetric distances).
        sz = (self.k_zeta, self.k_zeta)
        knn_tri_upper_inds = [np.ravel_multi_index((i, j), sz) for i in range(self.k_zeta) for j in range(i + 1, self.k_zeta)]



        patch_dat_tgt = np.zeros((1, patch_vol), dtype=np.float)

        # Loop over positive mask voxels:
        for n, tgt_ind in enumerate(tgt_inds):

            # Indices of centres of patches in neighbourhood.
            nbhd_inds = tgt_ind + nbhd_offsets

            # Indices of all patch voxels in nbhd.
            patch_inds[:] = nbhd_inds + patch_offsets

            # Pull patch data from reference array using raveled patch indices
            # Place in the remaining rows of patch_dat

            patch_dat_refs_ravel[:] = refs_flat[patch_inds_ravel]


            # pull patch data from target array.
            # Place it in the first row of patch_dat
            # Using the raveled version.

            patch_inds_tgt = tgt_ind + patch_offsets

            patch_dat_tgt[:] = tgt_flat[patch_inds_tgt]



            # Get squared Euclidean distances between target patch and all reference patches.
            d_tgt2refs = euclidean_distances(patch_dat_tgt, patch_dat_refs).ravel()


            # Get the k-nearest reference patches to the target patch.
            ix = np.argpartition(d_tgt2refs, self.k_zeta)[:self.k_zeta]

            d_tgt2refs = d_tgt2refs[ix]


            # Distances within the knn ref patches.
            d_refs2refs = euclidean_distances(patch_dat_refs[ix, :]).ravel()[knn_tri_upper_inds]



            out_dat_flat[tgt_ind] = (np.mean(d_tgt2refs) - np.mean(d_refs2refs)) / np.std(d_refs2refs)

            if n % 1000 == 0:
                print(n//1000)



        print('Done.')


        out_img = nib.Nifti1Image(out_dat, self.tgt.affine, self.tgt.header)

        nib.save(out_img, self.output)

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
