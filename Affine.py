"""
@author: jscs
"""
import os
import time
import nibabel as nib
import numpy as np
from os.path import basename
from dipy.viz import regtools
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)


def affine_reg(static_path, moving_path):
    """
    :param static_path:
    :param moving_path:
    :return:
    """
    t0_time = time.time()

    print('-->Applying affine reg over', basename(moving_path), 'based on', basename(static_path))

    static_img = nib.load(static_path)
    static = static_img.get_data()
    static_grid2world = static_img.affine

    moving_img = nib.load(moving_path)
    moving = np.array(moving_img.get_data())
    moving_grid2world = moving_img.affine

    print('---> I. Translation of the moving image towards the static image')

    print('---> Resembling the moving image on a grid of the same dimensions as the static image')

    identity = np.eye(4)
    affine_map = AffineMap(identity, static.shape, static_grid2world,  moving.shape, moving_grid2world)
    resampled = affine_map.transform(moving)

    regtools.overlay_slices(static, resampled, None, 0,
                            "Static", "Moving", "resampled_0.png")
    regtools.overlay_slices(static, resampled, None, 1,
                            "Static", "Moving", "resampled_1.png")
    regtools.overlay_slices(static, resampled, None, 2,
                            "Static", "Moving", "resampled_2.png")

    print('---> Aligning the centers of mass of the two images')

    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)

    print('---> We can now transform the moving image and draw it on top of the static image')

    transformed = c_of_mass.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_com_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_com_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_com_2.png")

    print('---> II. Refine  by looking for an affine transform')

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    """
    Now we go ahead and instantiate the registration class with the configuration
    we just prepared
    """
    print('---> Computing Affine Registration (non-convex optimization)')

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)

    transformed = translation.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_trans_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_trans_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_trans_2.png")

    print('--->III. Refining with a rigid transform')

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    transformed = rigid.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_rigid_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_rigid_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_rigid_2.png")

    print('--->IV. Refining with a full afine transform (translation, rotation, scale and shear)')

    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=starting_affine)

    print('---> Results in a slight shear and scale')

    transformed = affine.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_affine_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_affine_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_affine_2.png")

    name = os.path.splitext(basename(moving_path))[0] + '_affine_reg'
    nib.save(nib.Nifti1Image(transformed, np.eye(4)), name)
    t1_time = time.time()
    total_time = t1_time-t0_time
    print('Total time:' + str(total_time))
    return transformed
