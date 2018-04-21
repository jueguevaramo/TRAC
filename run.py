from Paths import *
from Basics import data
from Basics import otsu
from Basics import resli
from Basics import Nonlocal
from Basics import DTImodel
from Basics import gtab

resli(In.img)
Nonlocal("/home/jueguevaramo/Escritorio/Tractos/Reslice.nii")
otsu("/home/jueguevaramo/Escritorio/Tractos/Nonlocal.nii")
Imgpath="/home/jueguevaramo/Escritorio/Tractos/OtsuBoMask.nii"
Imgmask="/home/jueguevaramo/Escritorio/Tractos/OtsuMask.nii"
DTImodel(Imgpath,Imgmask,In.bval,In.bvec)