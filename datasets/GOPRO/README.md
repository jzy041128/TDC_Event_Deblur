# GoPro with scer

We synthetic events using original [GoPro dataset][gopro_website] 
and stored them as scer (Symmetric Cumulative Event Representation).


### Download

Original GoPro dataset can be downloaded [here][gopro_website].
GoPro dataset with events can be downloaded through the associated [website][efnet_website].


### Dataset Structure

The directory structure of GoPro is as follows:
```
GoPro/split/xxx.h5`
```

 - `split`     'train', or 'test'.
 - `xxx`       the sequence name.

### H5 file Structure

 - `images/image{:09d}` The blurry images
 - `sharp_images/image{:09d}` The ground truth sharp images
 - `voxels/voxel{:09d}` The events in the format of SCER (refer to the paper).

### Citation

If you use GoPro in your work, please cite the paper:

```
@InProceedings{Nah_2017_CVPR,
  author = {Nah, Seungjun and Kim, Tae Hyun and Lee, Kyoung Mu},
  title = {Deep Multi-Scale Convolutional Neural Network for Dynamic Scene Deblurring},
  booktitle = {CVPR},
  month = {July},
  year = {2017}
}
```

If you use GoPro with events in your work, please consider cite the paper:
```
@inproceedings{sun2022event,
      author = {Sun, Lei and Sakaridis, Christos and Liang, Jingyun and Jiang, Qi and Yang, Kailun and Sun, Peng and Ye, Yaozu and Wang, Kaiwei and Van Gool, Luc},
      title = {Event-Based Fusion for Motion Deblurring with Cross-modal Attention},
      booktitle = {European Conference on Computer Vision (ECCV)},
      year = 2022
      }
```

### Contact

Please feel free to contact us with any questions or comments:

[Lei Sun][personal_page], leo_sun [at] zju.edu.cn

[Project website][project_website]

[gopro_website]: <https://seungjunnah.github.io/Datasets/gopro>
[efnet_website]: <https://github.com/AHupuJR/EFNet>
[license_link]: <https://creativecommons.org/licenses/by/4.0/>
[project_website]: <https://ahupujr.github.io/EFNet/>
[personal_page]: <https://ahupujr.github.io/>