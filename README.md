# MedGA

MedGA is a novel evolutionary method based on Genetic Algorithms for the enhancemnet of bimodal biomedical images.

MedGA tackles the complexity of the enhancement problem by exploiting Genetic Algorithms to improve the appearance and the visual quality of images characterized by a bimodal gray level intensity histogram, by strengthening their two underlying sub-distributions.
This novel medical image enhancement technique is a promising solution suitable for medical expert systems.

  1. [Reference](#ref) 
  2. [Required library](#lib) 
  3. [Input parameters](#inp)
  4. [Data](#data)
  5. [License](#lic)
  6. [Contacts](#cont)
  
## <a name="ref"></a>Reference ##

A detailed description of MedGA, as well as a complete experimental comparison with standard state-of-the-art algorithm for image enhancement by using the dataset described below ([Data](#data)), can be found in:

## <a name="lib"></a>Required library ##

MedGA has been developed in Python and tested on Ubuntu Linux, MacOS X and Windows.

MedGA exploits the following libraries:
- `numpy`
- `scipy`
- `matplotlib`
- `mpi4py`, which provides bindings of the Message Passing Interface (MPI) specifications for Python.

The sequential version has been developed to analyze a single medical image, while the parallel version is based on a Master-Slave paradigm employing `mpi4py` to leverage High-Performance Computing (HPC) resources.
The parallel version has been implementated to perform the enhancement of multiple images (or slices in the case of tomography image stack analysis) in a parallel fashion.

## <a name="inp"></a>Input parameters ##

## <a name="data"></a>Data ##

### Dataset ###

## <a name="lic"></a>License ##

MedGA is licensed under the terms of the GNU GPL v3.0

## <a name="cont"></a>Contacts ##

For questions or support, please contact <andrea.tangherloni@disco.unimib.it>
and/or <leonardo.rundo@disco.unimib.it>
