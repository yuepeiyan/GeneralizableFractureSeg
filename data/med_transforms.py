import numpy as np
from monai import transforms
from torch import scalar_tensor, zero_


class ConvertToMultiChannelBasedOnBratsClassesd(transforms.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.concatenate(result, axis=0).astype(np.float32)
        return d


def get_scratch_train_transforms(args):
    if args.dataset == 'tibial_plateau':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                # transforms.Orientationd(keys=["image", "label"],
                #                         axcodes="RAS"),
                transforms.NormalizeIntensityd(keys=["image"], channel_wise=False),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),

                transforms.RandSpatialCropSamplesd(keys=["image", "label"],
                                                   roi_size=(args.roi_x, args.roi_y, args.roi_z),
                                                   random_size=False,
                                                   num_samples=args.num_samples),
                transforms.RandFlipd(keys=["image", "label"],
                                     prob=args.RandFlipd_prob,
                                     spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"],
                                     prob=args.RandFlipd_prob,
                                     spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"],
                                     prob=args.RandFlipd_prob,
                                     spatial_axis=2),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'pelvic':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                # transforms.Orientationd(keys=["image", "label"],
                #                         axcodes="RAS"),
                transforms.NormalizeIntensityd(keys=["image"], channel_wise=False),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),

                transforms.RandSpatialCropSamplesd(keys=["image", "label"],
                                                   roi_size=(args.roi_x, args.roi_y, args.roi_z),
                                                   random_size=False,
                                                   num_samples=args.num_samples),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    else:
        raise ValueError(f"{args.dataset} is not supported")
    return train_transform


def get_mae_pretrain_transforms(args):
    if args.datasetin in ['tibial_plateau', 'pelvic']:
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.AddChanneld(keys=["image"]),
                transforms.NormalizeIntensityd(keys=["image"], channel_wise=False),

                transforms.RandSpatialCropSamplesd(keys=["image"],
                                                   roi_size=(args.roi_x, args.roi_y, args.roi_z),
                                                   random_size=False,
                                                   num_samples=args.num_samples),
                transforms.RandFlipd(keys=["image"],
                                     prob=args.RandFlipd_prob,
                                     spatial_axis=0),
                transforms.RandFlipd(keys=["image"],
                                     prob=args.RandFlipd_prob,
                                     spatial_axis=1),
                transforms.RandFlipd(keys=["image"],
                                     prob=args.RandFlipd_prob,
                                     spatial_axis=2),
                transforms.ToTensord(keys=["image"]),
            ]
        )
    else:
        raise ValueError(f"{args.dataset} is not supported")
    return train_transform


def get_val_transforms(args):
    if args.dataset == 'tibial_plateau':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.NormalizeIntensityd(keys=["image"], channel_wise=False),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),

                transforms.RandSpatialCropSamplesd(keys=["image", "label"],
                                                   roi_size=(args.roi_x, args.roi_y, args.roi_z),
                                                   random_size=False,
                                                   num_samples=args.num_samples),
                transforms.RandFlipd(keys=["image", "label"],
                                     prob=args.RandFlipd_prob,
                                     spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"],
                                     prob=args.RandFlipd_prob,
                                     spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"],
                                     prob=args.RandFlipd_prob,
                                     spatial_axis=2),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'pelvic':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.NormalizeIntensityd(keys=["image"], channel_wise=False),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),

                transforms.RandSpatialCropSamplesd(keys=["image", "label"],
                                                   roi_size=(args.roi_x, args.roi_y, args.roi_z),
                                                   random_size=False,
                                                   num_samples=args.num_samples),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    else:
        raise ValueError(f"{args.dataset} is not supported")
    return val_transform


class Resize():
    def __init__(self, scale_params):
        self.scale_params = scale_params

    def __call__(self, img):
        scale_params = self.scale_params
        shape = img.shape[1:]
        assert len(scale_params) == len(shape)
        spatial_size = []
        for scale, shape_dim in zip(scale_params, shape):
            spatial_size.append(int(scale * shape_dim))
        transform = transforms.Resize(spatial_size=spatial_size, mode='nearest')
        # import pdb
        # pdb.set_trace()
        return transform(img)


def get_post_transforms(args):
    if args.dataset in ['tibial_plateau', 'pelvic']:
        if args.test:
            post_pred = transforms.Compose([transforms.EnsureType(),
                                            # Resize(scale_params=(args.space_x, args.space_y, args.space_z)),
                                            transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = transforms.Compose([transforms.EnsureType(),
                                             # Resize(scale_params=(args.space_x, args.space_y, args.space_z)),
                                             transforms.AsDiscrete(to_onehot=args.num_classes)])
        else:
            post_pred = transforms.Compose([transforms.EnsureType(),
                                            transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = transforms.Compose([transforms.EnsureType(),
                                             transforms.AsDiscrete(to_onehot=args.num_classes)])
    else:
        raise ValueError(f"{args.dataset} is not supported")
    return post_pred, post_label
