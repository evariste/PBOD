import numpy as np

from pyzeta import get_patch_offsets

print('===================================')
print('3D')
print('===================================')


sz = (4,5,6)

n_vox = int(np.prod(sz))
dat_flat = np.arange(n_vox)

dat_multi = np.reshape(dat_flat, sz)

idx_offsets = get_patch_offsets(1, sz)

cent_idx_flat = 40
cent_idx_multi = np.argwhere(dat_multi == cent_idx_flat)
ci, cj, ck = cent_idx_multi[0]

patch_range_multi = np.ix_(range(ci-1,ci+2), range(cj-1,cj+2), range(ck-1,ck+2))

patch_dat_multi = dat_multi[patch_range_multi]

patch_dat_flat = dat_flat[idx_offsets + cent_idx_flat]

print(patch_dat_flat)

print(patch_dat_multi)

print(patch_dat_flat.shape)

print(patch_dat_multi.shape)



print('===================================')
print('4D')
print('===================================')



# 4D 'image'
n_frames = 2

sz = (4,5,6,n_frames)

n_vox = int(np.prod(sz))
dat_flat = np.arange(n_vox)

dat_multi = np.reshape(dat_flat, sz)

idx_offsets = get_patch_offsets(1, sz)

cent_idx_flat = 100
cent_idx_multi = np.argwhere(dat_multi == cent_idx_flat)
ci, cj, ck, ct = cent_idx_multi[0]

patch_range_multi = np.ix_(range(ci-1,ci+2),
                           range(cj-1,cj+2),
                           range(ck-1,ck+2),
                           range(n_frames))

patch_dat_multi = dat_multi[patch_range_multi]

patch_dat_flat = dat_flat[idx_offsets + cent_idx_flat]

print(patch_dat_flat)

print(patch_dat_multi)

print(patch_dat_flat.shape)

print(patch_dat_multi.shape)


