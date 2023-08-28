## Normal Augmentations+

This notebook provides the code for training a network with the following augmentation settings: <br>
* Geometric Augmentations (e.g., Horizontal Flip, Rotation, etc.)
* Photometric Augmentations (e.g., Autocontrast, Equalization)
* Downsampling
* Common Corruptions
* Amplitude-Phase Recombination
<br><br>

## Author Notes: 
- This code is heavily based on the Amplitude-Phase Recombination (APR) because it is also used as one of the augmentations. It is only natural to leave their code intact to avoid messing up the result. Other augmentations are inserted in this code. For citations, refer to the respective links in the references.
<br><br>

## Usage Instructions: 
1. Install the anaconda environment -> environment.yml 
    * conda env create -f environment.yaml
2. Open the main.ipynb notebook
<br><br>

## References:
* Amplitude-Phase Recombination: https://github.com/iCGY96/APR
* Common Corruptions: https://github.com/bethgelab/imagecorruptions
