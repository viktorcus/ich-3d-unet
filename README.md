# ich-3d-unet
Project for Undergraduate Senior Thesis, B.S. Computer Science, Spring 2022. Segmentation of Intracranial Hemorrhage using a 3D U-Net.

The main code to run the network is in 3d_model.py, with helping files in the model directory.
The ICH image data should be stored in directories named npy-img and npy-mask, under the main directory. These files contain cleaned and normalized data for the ICH CT scans, stored in .npy format.
The CT Scans are from the dataset found at https://physionet.org/content/ct-ich/1.3.1/.

Citations for dataset:
Hssayeni, M. (2020). Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation (version 1.3.1). PhysioNet. https://doi.org/10.13026/4nae-zg36.
Hssayeni, M. D., Croock, M. S., Salman, A. D., Al-khafaji, H. F., Yahya, Z. A., & Ghoraani, B. (2020). Intracranial Hemorrhage Segmentation Using A Deep Convolutional Model. Data, 5(1), 14.
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.
