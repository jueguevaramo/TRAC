import nibabel as nib
import numpy as np
from dipy.align.reslice import reslice
from dipy.segment.mask import median_otsu
from dipy.denoise.nlmeans import nlmeans
from Affine import (c_of_mass,
                    translation,
                    rigid)
from dipy.align.imaffine import (MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import AffineTransform3D


def resli(ImgPath, newzooms=(2., 2., 2.)):
    """
    Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    int
        Description of return value

    """
    img = nib.load(ImgPath)
    data = img.get_data()
    affine = img.affine
    zooms = img.header.get_zooms()[:3]  # poner condicion de zooms
    data, affine = reslice(data, affine, zooms, newzooms)
    return nib.Nifti1Image(data, affine)


def otsu(img, median=4, pas=4, keep=False):
    """
    Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    int
        Description of return value

    """
    b0_mask, mask = median_otsu(img.get_data(), median, pas)
    b0_mask = nib.nifti1Nifti1Image(b0_mask, img.affine)
    mask = nib.Nifti1Image(mask, img.affine)
    return b0_mask, mask


def Nonlocal(img, keep=False, filt=100):
    """
    Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    int
        Description of return value

    """
    data = img.get_data()
    if (len(data.shape) == 3):
        mask = data > filt
    else:
        mask = data[..., 1] > filt
    data2 = data  # Preguntar
    sigma = np.std(data2[~mask])
    den = nlmeans(data2, sigma=sigma, mask=mask)
    return nib.Nifti1Image(den.astype(np.float32), img.affine)


def affine_4Dregistration(moving_img, static_img,
                          nbins=32,
                          sampling_prop=None,
                          metric='MI',
                          pipeline=[c_of_mass, translation, rigid],
                          level_iters=[10000, 1000, 100],
                          sigmas=[3.0, 1.0, 0.0],
                          factors=[4, 2, 1],
                          params0=None):
    """
    Affine_regtistration := Find the affine registration of a moving (4-D)
    based on an static (3-D).

    Parameters
    ----------
    moving_img: Nifti1Image (4-D)
    static_img: Nifti1Image (3-D)

    Returns
    -------
    Nifti1Image (4-D)
    """
    print('--> Affine registration')
    # Define the Affine registration object we'll use with the chosen metric:
    affine_metric_dict = {'MI': MutualInformationMetric}
    use_metric = affine_metric_dict[metric](nbins, sampling_prop)
    affreg = AffineRegistration(metric=use_metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)
    static = static_img.get_data()
    data_moving = moving_img.get_data()
    moving = data_moving[..., 0]
    # Bootstrap this thing with the identity:
    starting_affine = np.eye(4)
    # Go through the selected transformation:
    for func in pipeline:
        transformed, starting_affine = func(moving, static,
                                            moving_img.affine,
                                            static_img.affine,
                                            affreg, starting_affine,
                                            params0)
    transform = AffineTransform3D()
    affine = affreg.optimize(static, moving, transform, params0,
                             moving_img.affine, static_img.affine,
                             starting_affine=starting_affine)

    gradientDirections = data_moving.shape[-1]
    newData = np.zeros(static.shape + (int(gradientDirections),))
    for index in range(gradientDirections):
        print(index/gradientDirections)
        newData[:, :, :, index] = affine.transform(data_moving[:, :, :, index])

    return nib.Nifti1Image(newData, static_img.affine)
