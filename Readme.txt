Name: Yu Yue
Matriculation number: G2202151A
CodaLab username: YuYue525

--- Files submitted ---
GitHub:
1. report.pdf : short report in pdf format
2. Readme.txt : readme file
3. mmediting_v1.0: the folder contains msrresnet train and test configration file
4. mmediting_v0.0: the folder contains srgan and its msrresnet configration file, train and test script
5. CodaLab.png : Screenshot on CodaLab of the score achieved
6. test_real_results: contains results of the six real test images
7. model_best.txt: contains the link access to the best model and other files as follows:
OneDrive:
8. results_best.zip (in OneDrive): best predicted HQ images on the 400 test images
9. model_best.pth (in OneDrive): the SRResNet model checkpoint with the highest PSNR
10. model_srgan.pth (in OneDrive): the SRGAN model to generate natural images


--- Third-party libraries ---
1. torch
2. torchvision
3. mmediting 0.0
4. mmediting 1.0

--- Script to run ---
(please make sure that the corresponding versions of mmediting packages are already installed)
1. train and validate the msrresnet model
  cd mmediting_v1.0
  python mmediting/tools/train.py msrresnet.py

2. train and test the srgan model
  cd mmediting_v0.0
  python train.py
  python test.py
(please make sure the configurations in the train.py and test.py are all correct)
  
--- Links ---
1. GitHub: https://github.com/YuYue525/AI6126_project_2
2. model download: https://entuedu-my.sharepoint.com/:f:/g/personal/yyu025_e_ntu_edu_sg/EvddorSie99OooVPHmhOWy0B4PvVMfrQiVcnQsw4OpkFpg?e=Oiiiz2
