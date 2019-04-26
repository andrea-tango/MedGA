# MedGA

MedGA is a novel evolutionary method based on Genetic Algorithms for the enhancemnet of bimodal biomedical images.

MedGA tackles the complexity of the enhancement problem by exploiting Genetic Algorithms (GAs) to improve the appearance and the visual quality of images characterized by a bimodal gray level intensity histogram, by strengthening their two underlying sub-distributions.
This novel medical image enhancement technique is a promising solution suitable for medical expert systems.

  1. [Reference](#ref) 
  2. [Required libraries](#lib) 
  3. [Input parameters](#inp)
  4. [Data](#data)
  5. [License](#lic)
  6. [Contacts](#cont)
  
## <a name="ref"></a>Reference ##

A detailed description of MedGA, as well as a complete experimental comparison against state-of-the-art image enhancement techniques by using the dataset described below ([Data](#data)), can be found in:

- Rundo L., Tangherloni A., Nobile M.S., Militello C., Besozzi D., Mauri G., and Cazzaniga P.: "MedGA: a novel evolutionary method for image enhancement in medical imaging systems", Expert Systems with Applications, 119, 387-399, 2019. doi: 10.1016/j.eswa.2018.11.013


MedGA has been applied as a preprocessing step of a novel evolutionary framework for image enhancement, segmentation, and quantification. This framework has been applied to different clinical scenarios involving bimodal MR image analysis, as described in:

- Rundo L., Tangherloni A., Cazzaniga P., Nobile M.S., Russo G., Gilardi M.C., Vitabile S., Mauri G., Besozzi D., and Militello, C.: "A novel framework for MR image segmentation and quantification by using MedGA", Computer Methods and Programs in Biomedicine, 2019. doi: 10.1016/j.cmpb.2019.04.016

## <a name="lib"></a>Required library ##

MedGA has been developed in Python 2.7 and tested on Ubuntu Linux, MacOS X and Windows.

MedGA exploits the following libraries:
- `numpy`
- `scipy`
- `matplotlib`
- `pillow`
- `mpi4py`, which provides bindings of the Message Passing Interface (MPI) specifications for Python.

The input images are characterized by an undelying bimodal histogram (i.e., intensity level distribution).
Possibly, these images might be previously masked and cropped images (according to a bounding region containing the Region of Interest).

The sequential version has been developed to analyze a single medical image, while the parallel version is based on a Master-Slave paradigm employing `mpi4py` to leverage High-Performance Computing (HPC) resources.
The parallel version has been implementated to perform the enhancement of multiple images (or slices in the case of tomography image stack analysis) in a distributed fashion.

Notice that `MPI` (such as mpich on Linux OS, Open MPI on MacOS X, and MS-MPI on Windows) and `mpi4py` are not strictly required. The sequential version of MedGA can be used to analyze a folder containing multiple images.

## <a name="inp"></a>Input parameters ##

In oder to run MedGA, the following parameters must be provided:

- `-f` to specify the input folder containing the images to process (in TIFF/TIF, PNG, JPEG/JPG format);
- `-i` to specify the input image to process (in TIFF/TIF, PNG, JPEG/JPG format);

Note that `-f` and `-i` are mutually exclusive. If both of them are used, MedGA stops requiring either `-f` or `-i`.
  
Optional parameters could be provided:

- `-o` to specify the output folder (default: output);
- `-p` to specify the population size (number of individuals) of the GA (default: 100);
- `-g` to specify the number of generations of the GA (default: 100);
- `-s` to specify the selection strategy. The available strategies are: wheel, ranking, and tournament (default: tournament);
- `-c` to specify the crossover rate (default: 0.9, i.e., 90%);
- `-m` to specify the mutation rate (default: 0.01, i.e.,  1%);
- `-k` to specify the number of individuals used in each tournament. This parameter is used if and only if the tournament selection strategies is used (default: 20);
- `-e` to specify the number of best individuals of the current generation that will survive in the next generation without modifications (default: 1);
- `-d` to enable the distributed version of `MedGA` by using `MPI` and `mpi4py` (default: False);
- `-t` to specify the number of cores exploited by distributed version of `MedGA` (default: 4);
- `-v` to enable the verbose modality (default: False).

For example, `MedGA` can be executed with the following command:

    python MedGA.py -i fibroids/01.tiff
    python MedGA.py -f fibroids

The **former** will save the following files in the **_output/01_** folder:
- _fitness_ contains the fitness value of the best individual for each generation;
- _information_ contains the GA settings used to perform the optimization. Note that the number of genes corresponds to the number of pixels composing the analyzed image;
- _matrixBest_ contains the final best image as `TSV` file;
- _terms_ contains the values of the 3 different terms composing the fitness function for each generation;
- _threshold_ contains the optimal threshold for each generation.

The **_images_** subfolder contains the original image, the best image after the GA initialization and the final best image.

The **latter** will save the described files and subfolder for each image in the **_fibroids_** folder.

By running `python MedGA.py` without specifying any parameter (or using `-h`), all the above parameters will be listed.

## <a name="data"></a>Data ##

### Dataset ###

## <a name="lic"></a>License ##

MedGA is licensed under the terms of the GNU GPL v3.0

## <a name="cont"></a>Contacts ##

For questions or support, please contact <andrea.tangherloni@disco.unimib.it> (<at860@cam.ac.uk>)
and/or <leonardo.rundo@disco.unimib.it> (<lr495@cam.ac.uk>).
