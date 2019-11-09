import os
import argparse

from defaults import defaults
from pyzeta import PyZeta


###########################################################################

def usage():

    desc = 'Generate a zeta map for an image given a reference set.'
    parser = argparse.ArgumentParser(description=desc)


    help_text = 'Target image for which the map is needed.'
    parser.add_argument('target', type=str, help=help_text)

    help_text = 'Reference set, a folder of images or a 4D image.'
    parser.add_argument('ref_set', type=str, help=help_text)

    help_text = 'Output image ontaining zeta map.'
    parser.add_argument('output', type=str, help=help_text)

    help_text = ('Mask image defining the voxels where the zeta'
                 ' values are calculated.')
    parser.add_argument('-mask', type=str, help=help_text, default=None)

    n = defaults['k_zeta']
    help_text = ('Number of nearest neighbours, \'k\', to use in zeta'
                 ' calculation. Default is {:d}'.format(n))
    parser.add_argument('-k_zeta', type=int, help=help_text, default=n)

    n = defaults['patch_radius']
    help_text = ('Radius of patch (integer). Patch edge length is 2*R+1.'
                 ' Default is {:d}.'.format(n))
    parser.add_argument('-patch_radius', type=int, help=help_text, default=n)

    n = defaults['nbhd_radius']
    help_text = ('Neighbourhood search radius over which to search.'
                 ' Default is {:d}'.format(n))
    parser.add_argument('-nbhd_radius', type=int, help=help_text, default=n)

    help_text = ('Use Euclidean distance between patches instead of'
                 ' the default Mahalanobis distance.')
    parser.add_argument('-euclidean', action='store_true', help=help_text)

    help_text = 'Optional config file containing parameter choices.'
    parser.add_argument('-config_file', type=str, help=help_text, default=None)


    args = parser.parse_args()


    return args

###########################################################################

def main():

    args = usage()

    pyz = PyZeta(args)

    pyz.run()

    return 0

###########################################################################

if __name__ == '__main__':

    os.sys.exit(main())

