__author__ = 'Jrudascas'
import numpy as np
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)

from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
import pdb


def c_of_mass(moving, static, static_grid2world, moving_grid2world,
              reg, starting_affine, params0=None):
    transform = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)
    transformed = transform.transform(moving)
    return transformed, transform.affine


def translation(moving, static, static_grid2world, moving_grid2world,
                reg, starting_affine, params0=None):
    transform = TranslationTransform3D()
    translation = reg.optimize(static, moving, transform, params0,
                               static_grid2world, moving_grid2world,
                               starting_affine=starting_affine)

    return translation.transform(moving), translation.affine


def rigid(moving, static, static_grid2world, moving_grid2world,
          reg, starting_affine, params0=None):
    transform = RigidTransform3D()
    rigid = reg.optimize(static, moving, transform, params0,
                         static_grid2world, moving_grid2world,
                         starting_affine=starting_affine)
    return rigid.transform(moving), rigid.affine


def affine(moving, static, static_grid2world, moving_grid2world,
           reg, starting_affine, params0=None):
    transform = AffineTransform3D()
    affine = reg.optimize(static, moving, transform, params0,
                          static_grid2world, moving_grid2world,
                          starting_affine=starting_affine)

    return affine.transform(moving), affine.affine


def affine_registration(moving, static,
                        moving_grid2world=None,
                        static_grid2world=None,
                        nbins=32,
                        sampling_prop=None,
                        metric='MI',
                        pipeline=[c_of_mass, translation, rigid, affine],
                        level_iters=[10000, 1000, 100],
                        sigmas=[3.0, 1.0, 0.0],
                        factors=[4, 2, 1],
                        params0=None):
    """
    Find the affine transformation between two 3D images.data()
    """
    print('--> Affine registration')
    # Define the Affine registration object we'll use with the chosen metric:
    affine_metric_dict = {'MI': MutualInformationMetric}
    use_metric = affine_metric_dict[metric](nbins, sampling_prop)
    affreg = AffineRegistration(metric=use_metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    # Bootstrap this thing with the identity:
    starting_affine = np.eye(4)
    # Go through the selected transformation:
    for func in pipeline:
        transformed, starting_affine = func(moving, static,
                                            static_grid2world,
                                            moving_grid2world,
                                            affreg, starting_affine,
                                            params0)
    return transformed, starting_affine


pdb.set_trace()
from Paths import *
import nibabel as nib
moving = '/home/jueguevaramo/github/Tractos/B0_image.nii'
static = In.canonicalt1
moving = nib.load(moving).get_data()
static = nib.load(static).get_data()

transfomed, affine = affine_registration(moving, static)

pdb.set_trace()
'''
def affine_registration4d(moving, static,
                          moving_grid2world=None,
                          static_grid2world=None,
                          nbins=32,
                          sampling_prop=None,
                          metric='MI',
                          pipeline=[c_of_mass, translation, rigid, affine],
                          level_iters=[10000, 1000, 100],
                          sigmas=[3.0, 1.0, 0.0],
                          factors=[4, 2, 1],
                          params0=None):


    print('--> Affine registration')
    # Define the Affine registration object we'll use with the chosen metric:
    affine_metric_dict = {'MI': MutualInformationMetric}
    use_metric = affine_metric_dict[metric](nbins, sampling_prop)
    affreg = AffineRegistration(metric=use_metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    # Bootstrap this thing with the identity:
    starting_affine = np.eye(4)
    # Go through the selected transformation:
    for func in pipeline:
        transformed, starting_affine = func(moving, static,
                                            static_grid2world,
                                            moving_grid2world,
                                            affreg, starting_affine,
                                            params0)

    return transformed, starting_affine
'''
