import cv2
import torch
import numpy as np

# Trim circle image
class TrimpsCircle:
    def __call__(self, image: torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()        
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (int(width // 2.01), int(height // 2.01))
        radius = min(center[0], center[1])  
        # img = cv2.circle(mask, center, radius, (255, 255, 255), -1)
        if image.shape[2] == 4:
            alpha_channel = image[:, :, 3]
            image_rgb = image[:, :, :3]
            masked_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
            masked_alpha = cv2.bitwise_and(alpha_channel, alpha_channel, mask=mask)
            result = cv2.merge([masked_rgb[:, :, 0], masked_rgb[:, :, 1], masked_rgb[:, :, 2], masked_alpha])
        else: result = cv2.bitwise_and(image, image, mask=mask)
        result = torch.tensor(result).permute(2,0,1)
        return result

class ScaleRadiusTransform:
    def __call__(self, image: torch.Tensor, scale=500):
        img = image.permute(1,2,0).numpy()
        a = self.scaleRadius(img, scale)
        b = np.zeros(a.shape)
        x = a.shape[1] / 2
        y = a.shape[0] / 2
        center_coordinates = (int(x), int(y))
        cv2.circle(b, center_coordinates, int(scale * 0.9), (1, 1, 1), -1, 8, 0)
        aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128) * b + 128 * (1 - b)
        aa = torch.tensor(aa).permute(2,0,1).int()
        return aa

    def scaleRadius(self, image, scale):
        k = image.shape[0]/2
        x = image[int(k), :, :].sum(1)
        r=(x>x.mean()/10).sum()/2
        if r == 0: r = 1
        s=scale*1.0/r
        return cv2.resize(image,(0,0),fx=s,fy=s)