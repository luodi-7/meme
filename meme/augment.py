import math
import os
from typing import Optional, Tuple, Union

import torch
from torchvision import transforms

from PIL import Image

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import str_to_interp_mode, str_to_pil_interp, RandomResizedCropAndInterpolation, \
    ResizeKeepRatio, CenterCropOrPad, RandomCropOrPad, TrimBorder, ToNumpy, MaybeToTensor, MaybePILToTensor
from timm.data.random_erasing import RandomErasing

def augment_meme(img_size: Union[int, Tuple[int, int]] = 224,
        scale: Optional[Tuple[float, float]] = None,
        ratio: Optional[Tuple[float, float]] = None,
        train_crop_mode: Optional[str] = None,
        color_jitter: Union[float, Tuple[float, ...]] = 0.4,
        color_jitter_prob: Optional[float] = None,
        force_color_jitter: bool = False,
        grayscale_prob: float = 0.,
        gaussian_blur_prob: float = 0.,
        auto_augment: Optional[str] = None,
        interpolation: str = 'random',
        mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
        re_prob: float = 0.,
        re_mode: str = 'const',
        re_count: int = 1,
        re_num_splits: int = 0,
        use_prefetcher: bool = False,
        normalize: bool = True,
        separate: bool = False,
        hflip = 0,
        vflip = 0
        ):

    train_crop_mode = train_crop_mode or 'rrc'
    assert train_crop_mode in {'rrc', 'rkrc', 'rkrr'}
    if train_crop_mode in ('rkrc', 'rkrr'):
        # FIXME integration of RKR is a WIP
        scale = tuple(scale or (0.8, 1.00))
        ratio = tuple(ratio or (0.9, 1/.9))
        primary_tfl = [
            ResizeKeepRatio(
                img_size,
                interpolation=interpolation,
                random_scale_prob=0.5,
                random_scale_range=scale,
                random_scale_area=True,  # scale compatible with RRC
                random_aspect_prob=0.5,
                random_aspect_range=ratio,
            ),
            CenterCropOrPad(img_size, padding_mode='reflect')
            if train_crop_mode == 'rkrc' else
            RandomCropOrPad(img_size, padding_mode='reflect')
        ]
    else:
        scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
        ratio = tuple(ratio or (3. / 4., 4. / 3.))  # default imagenet ratio range
        primary_tfl = []
    if hflip > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str)
        # color jitter is typically disabled if AA/RA on,
        # this allows override without breaking old hparm cfgs
        disable_color_jitter = not (force_color_jitter or '3a' in auto_augment)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = str_to_pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]

    if color_jitter is not None and not disable_color_jitter:
        # color jitter is enabled when not using AA or when forced
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        if color_jitter_prob is not None:
            secondary_tfl += [
                transforms.RandomApply([
                        transforms.ColorJitter(*color_jitter),
                    ],
                    p=color_jitter_prob
                )
            ]
        else:
            secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    if grayscale_prob:
        secondary_tfl += [transforms.RandomGrayscale(p=grayscale_prob)]

    if gaussian_blur_prob:
        secondary_tfl += [
            transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=23),  # hardcoded for now
                ],
                p=gaussian_blur_prob,
            )
        ]

    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    elif not normalize:
        # when normalize disable, converted to tensor without scaling, keeps original dtype
        final_tfl += [MaybePILToTensor()]
    else:
        final_tfl += [
            MaybeToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std),
            ),
        ]
        if re_prob > 0.:
            final_tfl += [
                RandomErasing(
                    re_prob,
                    mode=re_mode,
                    max_count=re_count,
                    num_splits=re_num_splits,
                    device='cpu',
                )
            ]

    if separate:
        return transforms.Compose(secondary_tfl), transforms.Compose(final_tfl)
    else:
        return transforms.Compose(primary_tfl + secondary_tfl)


transform_meme = augment_meme(
    img_size=224,
    color_jitter=0.4,
    grayscale_prob=0.2,
    gaussian_blur_prob=0.5,
    auto_augment='rand-m9-mstd0.5',
    normalize=True
)


# 遍历输入目录下的所有文件
input_dir = '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/'
N = 3
for filename in os.listdir(input_dir):
    # 检查文件是否为图片文件
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) and filename.lower()[-5]==')':
        # 构建输入文件的完整路径
        input_path = os.path.join(input_dir, filename)

        # 打开图片
        try:
            image = Image.open(input_path).convert('RGB')
        except Exception as e:
            print(e)
            continue

        # 对图片进行 N 次增强
        for i in range(N):
            # 应用变换
            try:
                transformed_image = transform_meme(image)
                # 将张量转换回 PIL 图像
                # transformed_image = transforms.ToPILImage()(transformed_image)

                # 构建输出文件的完整路径
                base_name, ext = os.path.splitext(filename)
                output_filename = f"{base_name}_{i}{ext}"
                output_path = os.path.join(input_dir, output_filename)

                # 保存变换后的图片
                transformed_image.save(output_path)
                print(f"Saved {output_path}")

            except Exception as e:
                print(e)
                continue