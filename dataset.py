import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import prepare_data
from albumentations.torch.functional import img_to_tensor


class RoboticsDataset(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train', problem_type=None, right_frames = False):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type
        self.right_frames = right_frames

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name, self.problem_type, self.right_frames)

        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        if self.mode == 'train':
            if self.problem_type == 'binary':
                return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                return img_to_tensor(image), torch.from_numpy(mask).long()
        else:
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, problem_type, right_frames):
    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = prepare_data.binary_factor
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = prepare_data.parts_factor
    elif problem_type == 'instruments':
        factor = prepare_data.instrument_factor
        mask_folder = 'instruments_masks'

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)
    if right_frames:
        mask = shift_label(mask, mask.shape[0], mask.shape[1])
        """
        print("Mask is: ")
        print(mask)
        print("With shape: ")
        print(mask.shape)
        print("Calling shift with shape 0 and 1: ")
        new_mask = shift_label(mask, mask.shape[0], mask.shape[1])
        print("Success!, diff in nonzero: ")
        print("old")
        print(np.nonzero(mask))
        print("new")
        print(np.nonzero(new_mask))
        """
    return (mask / factor).astype(np.uint8)
    

def shift_label(label, height, width):
  shift_i = 10
  shift_j = 13

  label_right = np.zeros((height, width))
  for i in range(height-1):
    for j in range(width-1):
      x_right_i = round(i + shift_i)
      x_right_j = round(j + shift_j)
      if x_right_i >= 0 and x_right_i < height:
        if x_right_j >= 0 and x_right_j < width:
          label_right[x_right_i, x_right_j] = label[i, j];
  return label_right
