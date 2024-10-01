from skimage.measure import label, regionprops, regionprops_table
import torchvision.transforms.v2.functional as TF
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.io import read_image
from PIL import Image
import numpy as np
import torch
import cv2
import os

class CP:
    def __call__(self, image: torch.Tensor):
        image_rgb = image.permute(1,2,0).numpy()
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(image_rgb)
        r_eq = cv2.equalizeHist(r)
        g_eq = cv2.equalizeHist(g)
        b_eq = cv2.equalizeHist(b)
        equalized_image_rgb = cv2.merge([r_eq, g_eq, b_eq])
        equalized_image_rgb = torch.tensor(equalized_image_rgb).permute(2,0,1)
        return equalized_image_rgb

class GussianBlurCV:
    def __call__(self, image: torch.Tensor):
        image = image.permute(1,2,0).numpy()
        blurred = cv2.GaussianBlur(image, (0, 0), 10)
        output_cv = cv2.addWeighted(image, 4, blurred, -4, 128)
        output_cv = torch.tensor(output_cv).permute(2,0,1)
        return output_cv

class GussianBlurTorch:
    def __call__(self, image: torch.Tensor):
        kernel_size = (39, 39)
        sigma = 100
        blurred = TF.gaussian_blur(image.unsqueeze(0), kernel_size=kernel_size, sigma=(sigma, sigma))
        output = 4 * image - 4 * blurred.squeeze(0) + 128
        output = torch.clamp(output, 0, 255)
        return output

class HSVTransform:
    def __call__(self, image: torch.Tensor):
        r, g, b = image[0], image[1], image[2]
        max_val, _ = torch.max(image, dim=0)
        min_val, _ = torch.min(image, dim=0)
        delta = max_val - min_val
        h = torch.zeros_like(max_val, dtype=torch.float32)
        s = torch.zeros_like(max_val, dtype=torch.float32)
        v = max_val.float()  #
        mask = delta != 0
        mask_r = (max_val == r) & mask
        mask_g = (max_val == g) & mask
        mask_b = (max_val == b) & mask
        h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
        h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2
        h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4
        h = h / 6.0
        s[mask] = delta[mask] / v[mask]
        s[~mask] = 0
        hsv_image = torch.stack([h, s, v], dim=0).int()
        return hsv_image

class SLAHE:
    def __call__(self, image: torch.Tensor):
        image_rgb = image.permute(1,2,0).numpy()
        r, g, b = cv2.split(image_rgb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        r_clahe = clahe.apply(r)
        g_clahe = clahe.apply(g)
        b_clahe = clahe.apply(b)
        clahe_image_rgb = cv2.merge((r_clahe, g_clahe, b_clahe))
        # clahe_image = cv2.cvtColor(clahe_image_rgb, cv2.COLOR_RGB2BGR)
        clahe_image_rgb = torch.tensor(clahe_image_rgb).permute(2, 0, 1)
        return clahe_image_rgb

class PolaLinear:
    def __call__(self, image):
        image = image.permute(1,2,0).cpu().numpy()
        center = (image.shape[1] // 2, image.shape[0] // 2)
        max_radius = np.sqrt((center[0] ** 2) + (center[1] ** 2))

        polar_image = cv2.linearPolar(image, center, max_radius, cv2.WARP_FILL_OUTLIERS)
        img = self.crop_(polar_image)
        img = torch.tensor(img).permute(2, 0, 1)
        return img

    def crop_(self, image):
        img_ = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(img_, 10, 255, cv2.THRESH_BINARY)
        labeled_image = label(thresh)
        regions = regionprops(labeled_image)
        for region in regions:
            if region.area >= 500:
                minr, minc, maxr, maxc = region.bbox
                # imgp = cv2.rectangle(imgp, (minr, minc), (maxr, maxc), (0, 255, 0), 2)
                image = image[int(minr):int(maxr), int(minc):int(maxc-10)]
        return image

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    image = 'path/to/image'
    img = read_image(image)
    tran = v2.Compose([
        # TrimpsCircle(),
        # CP(),
        # GussianBlurCV() ,
        # GussianBlurTorch(),
        # ScaleRadiusTransform(),
        # HSVTransform(),
        # SLAHE(),
        # PolaLinear(),
    ])
    img2 = tran(img)
    plt.imshow(img2.permute(1,2,0).numpy())