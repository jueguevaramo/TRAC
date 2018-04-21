import nibabel as nib
import numpy as np
from dipy.data import get_data
from dipy.io import read_bvals_bvecs
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.align.reslice import reslice
from dipy.segment.mask import median_otsu
from dipy.denoise.nlmeans import nlmeans
import dipy.reconst.dti as dti

def data(ImgPath):
    img=nib.load(ImgPath) #Muchas veces get_data
    return img.get_data()

def gtab(fbvalPath,fbvecPath):
    bvals, bvecs = read_bvals_bvecs(fbvalPath, fbvecPath)
    return (gradient_table(bvals, bvecs))

def resli(ImgPath):
    img=nib.load(ImgPath)
    data=img.get_data()
    affine=img.affine
    zooms=img.header.get_zooms()[:3]
    newzooms=(2.,2.,2.)
    data, affine = reslice(data, affine, zooms,newzooms)
    nib.save(nib.Nifti1Image(data, affine),"Reslice")
    return print("Reslice hecho")

def otsu(ImgPath):
    img = nib.load(ImgPath)
    data = img.get_data()
    b0_mask, mask = median_otsu(data, 3, 3)
    nib.save(nib.Nifti1Image(b0_mask.astype(np.float32), img.get_affine()),"OtsuBoMask")
    nib.save(nib.Nifti1Image(mask.astype(np.float32), img.get_affine()),"OtsuMask")
    return print("Otsu hecho")

def Nonlocal(ImgPath): #Preguntar! #No usan denoise images PCA
    img = nib.load(ImgPath)
    data = img.get_data()
    mask = data[..., 1] > 49
    data2 = data #Preguntar
    sigma = np.std(data2[~mask])
    den = nlmeans(data2, sigma=sigma, mask=mask)
    nib.save(nib.Nifti1Image(den.astype(np.float32), img.get_affine()),"Nonlocal")
    return print("Non local hecho")


def DTImodel(ImgPath,ImgMaskPath,bvalsPath,bvecsPath):
    img = nib.load(ImgPath)
    data = img.get_data()
    mask = nib.load(ImgMaskPath)
    mask = mask.get_data()
    bvals, bvecs = read_bvals_bvecs(bvalsPath,bvecsPath)
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask)
    evecs_img = nib.Nifti1Image(tenfit.evecs.astype(np.float32), img.get_affine())
    evals_img = nib.Nifti1Image(tenfit.evals.astype(np.float32), img.get_affine())
    nib.save(evecs_img, "evecs")
    nib.save(evals_img, "evals")

    return print("DTI model is done")
