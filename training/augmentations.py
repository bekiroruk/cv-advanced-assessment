
import albumentations as A

def get_train_transforms(img_size=640):
    """
    Assignment'te istenen güçlü augmentations için örnek pipeline.
    Şu an YOLO'nun kendi augment'leri kullanılacak,
    ama bu fonksiyon gerektiğinde custom dataloader ile entegre edilebilir.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), p=1.0),
            A.OneOf(
                [
                    A.MotionBlur(p=0.5),
                    A.GaussianBlur(p=0.5),
                ],
                p=0.3,
            ),
            A.Cutout(num_holes=8, max_h_size=img_size // 10, max_w_size=img_size // 10, p=0.3),
            A.ColorJitter(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )
