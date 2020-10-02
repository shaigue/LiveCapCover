# LiveCapCover

## Description:
This is a university (Technion, Israel Institute of technology) undergraduate project in image processing.
It is a inspired from the **LiveCap** paper listed in the credits. We implement a part of the paper, and we deviate 
from the original paper in some aspects. 
 
## Authors:
* Izo Sakallah [email](izo.sakallah@gmail.com)
* Shai Guendelman [email](shaigue@gmail.com)

## Credits:
1. to the original authors of the LiveCap paper, the project page is in [here](https://gvv.mpi-inf.mpg.de/projects/LiveCap/)
2. Instead of [VNect](http://gvv.mpi-inf.mpg.de/projects/VNect/) we used [VIBE](https://is.mpg.de/publications/vibe-cvpr-2020), 
and their available code hosted in [here](https://github.com/mkocabas/VIBE)

## Details:

### running instructions

1. install the conda environment (of using any other package manager)
this is done first by installing miniconda https://docs.conda.io/en/latest/miniconda.html.
then using the "create from file option", after activating conda run:
`
conda env create -f environment.yml
`
you would create the `gip-vibe` environment. to learn more on conda read https://docs.conda.io/projects/conda/en/latest/index.html.
2. get the livecap data from their website: https://gvv-assets.mpi-inf.mpg.de/, you need to register,
after registration you need to look at the instruction on how to download.
you might use the header you copied with the curl tool, with the script lib/original/get_original.py,
and insert the copied header in the respected place. in the script you can see the files that are needed for download.
3. before running vibe, you need to make some preparation, and download their models.
use the lib/image_processing/vibe/prepare_data.py script to get it. it might do some path related bugs, 
just fix the paths there.
a small Note - in one of the packages that vibe uses `wget` command, or other linux shell commands that are not available on
windows. you might need to go to the buggy code and replace it with python platform agnostic code.
4. now after getting the video that you want, go to lib/data_utils/create_dataset_original.py, and create your dataset.
this will split your video into frames, then create the bounding boxes, vibe features, foreground masks.
5. then you can run the optimization on the model, lib/skeletal_pose_estimation/optimization2.py, to run it, then render it and enjoy the movies.

### Note:
The code is a little bit messy, so I would recommend using it as a reference, and taking parts of it, rather then making it work as is.
feel free to contact me if any question arises.



### directory structure

#### assets
contains static assets(not code) of the project, like the experiments results, camera calibration parameters and rigged models

#### datasets
here goes the frames, extracted keypoints, this should not be uploaded to github due to the large size.

#### lib
here is the code. it is separated into submodules.

#### tests
some visual tests(not automatic) for verifying that the modules run correctly.

#### old_src
this is old work, it is just there for reference, it is not needed for running the program
