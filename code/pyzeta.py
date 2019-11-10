import os
from defaults import defaults


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
