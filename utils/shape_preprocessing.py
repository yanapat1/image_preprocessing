import cv2
import torch
import numpy as np
from skimage.measure import label, regionprops
from torchvision.transforms import v2

# Resize and Crop the image: TrimpsCircle and CropPreprocess should use togather
class ResizeAndCrop:
    def __call__(self, image:torch.Tensor ,size=(1024,1024),size2=(400,400)):
        target_size = size
        if isinstance(image, torch.Tensor):
            image = image.permute(1,2,0).numpy()

        h, w = image.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
        output_image = np.full((target_size[1], target_size[0], 3), (0, 0, 0), dtype=np.uint8)
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2
        output_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
        # output_image
        img2 = cv2.cvtColor(output_image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)

        labeled_image = label(thresh)
        regions = regionprops(labeled_image)
        for region in regions:
            if region.area >= 1000:
                minr, minc, maxr, maxc = region.bbox
                minr, maxr = max(0, minr-4), min(output_image.shape[0] - 1, maxr+4)
                minc, maxc = max(0, minr-2), min(output_image.shape[1] - 1, maxc)
                output_image = output_image[int(minr):int(maxr), int(minc):int(maxc)]

        output_image = cv2.resize(output_image, size2)
        imageP = v2.ToPILImage()(output_image)
        imageT = v2.ToImage()(imageP)
        imageT = v2.ToDtype(torch.float32, scale=True)(imageT)
        
        return imageT

class TrimpsCircle:
    def __call__(self, image: torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()        
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (int(width // 2.01), int(height // 2.01))
        radius = min(center[0], center[1])  
        img = cv2.circle(mask, center, radius, (255, 255, 255), -1)
        if image.shape[2] == 4:
            alpha_channel = image[:, :, 3]
            image_rgb = image[:, :, :3]
            masked_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
            masked_alpha = cv2.bitwise_and(alpha_channel, alpha_channel, mask=mask)
            result = cv2.merge([masked_rgb[:, :, 0], masked_rgb[:, :, 1], masked_rgb[:, :, 2], masked_alpha])
        else: result = cv2.bitwise_and(image, image, mask=mask)
        result = torch.tensor(result).permute(2,0,1)
        return result

if __name__ == '__main__':
    pass
    # import matplotlib.pyplot as plt
    # from torchvision.io import read_image
    # from torchvision.transforms import v2

    # image = 'path/to/image'
    # img = read_image(image)
    # tran = v2.Compose([
    #     # TrimpsCircle(),
    # ])
    # img2 = tran(img)
    # plt.imshow(img2.permute(1,2,0).numpy())