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
from dipy.reconst.dti import color_fa, fractional_anisotropy, quantize_evecs
from dipy.data import get_sphere
from dipy.tracking.eudx import EuDX

def resli(ImgPath,keep=False,newzooms=(2.,2.,2.)): #Corregir newzooms entrada
    img=nib.load(ImgPath)
    data=img.get_data()
    affine=img.affine
    zooms=img.header.get_zooms()[:3] #poner condicion de zooms 
    data, affine = reslice(data, affine, zooms,newzooms)
    if keep:
        nib.save(nib.Nifti1Image(data, affine),"Reslice")
    return data, affine

def otsu(data,affine,median=4,pas=4,keep=False): #Corregir median, pas entrada 
    b0_mask, mask = median_otsu(data,median,pas)
    if keep:
        nib.save(nib.Nifti1Image(b0_mask.astype(np.float32), affine),"OtsuBoMask")
        nib.save(nib.Nifti1Image(mask.astype(np.float32), affine),"OtsuMask")
    return b0_mask, mask

def Nonlocal(data,affine,keep=False,filt=26): #Preguntar! #No usan denoise images PCA
    mask = data[..., 1] > filt
    data2 = data #Preguntar
    sigma = np.std(data2[~mask])
    den = nlmeans(data2, sigma=sigma, mask=mask)
    if keep:    
        nib.save(nib.Nifti1Image(den.astype(np.float32),affine),"Nonlocal")
    return den

def gtab(fbvalPath,fbvecPath):
    bvals, bvecs = read_bvals_bvecs(fbvalPath, fbvecPath)
    return (gradient_table(bvals, bvecs))

def DTImodel(data,affine,mask,gtab,keep=False):
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask)
    if keep:
        evecs_img = nib.Nifti1Image(tenfit.evecs.astype(np.float32), affine)
        evals_img = nib.Nifti1Image(tenfit.evals.astype(np.float32), affine)
        nib.save(evecs_img, "evecs")
        nib.save(evals_img, "evals")
    return tenfit.evals,tenfit.evecs

def DTImaps(ImgPath,Bvalpath,Bvecpath,tracto=True):
    data, affine=resli(ImgPath,keep=True)
    data= Nonlocal(data,affine,keep=True)
    b0_mask, mask=otsu(data,affine,keep=True)  #maask binary 
    evals,evecs=DTImodel(b0_mask,affine,mask,gtab(Bvalpath,Bvecpath))
    print('--> Calculando el mapa de anisotropia fraccional')
    FA = fractional_anisotropy(evals)
    FA[np.isnan(FA)] = 0
    nib.save(nib.Nifti1Image(FA.astype(np.float32), affine),"Mapa_anisotropia_fraccional")
    print('--> Calculando el mapa de anisotropia fraccional RGB')
    FA2 = np.clip(FA, 0, 1)
    RGB = color_fa(FA2, evecs)
    nib.save(nib.Nifti1Image(np.array(255 * RGB, 'uint8'), affine),"Mapa_anisotropia_fraccional RGB")
    print('--> Calculando el mapa de difusividad media')
    MD1 = dti.mean_diffusivity(evals)
    nib.save(nib.Nifti1Image(MD1.astype(np.float32), affine),"Mapa_difusividad_media")
    if tracto:
        sphere = get_sphere('symmetric724')
        peak_indices = quantize_evecs(evecs, sphere.vertices)

        eu = EuDX(FA.astype('f8'), peak_indices, seeds=500000, odf_vertices = sphere.vertices, a_low=0.15)
        tensor_streamlines = [streamline for streamline in eu]
        new_vox_sz = (2.,2.,2.)
        hdr = nib.trackvis.empty_header()
        hdr['voxel_size'] = new_vox_sz
        hdr['voxel_order'] = 'LAS'
        hdr['dim'] = FA.shape

        tensor_streamlines_trk = ((sl, None, None) for sl in tensor_streamlines)
        ten_sl_fname = "Tracto.trk"
        nib.trackvis.write(ten_sl_fname, tensor_streamlines_trk, hdr, points_space='voxel')
    return ("Done")
