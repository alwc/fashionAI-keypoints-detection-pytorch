import cv2
import numpy as np
import torch
import torch.nn.functional as F


class KeypointEncoder:

    def encode(self, keypoints, input_size, stride, hm_alpha, hm_sigma):
        '''For each image, create heatmaps and visibility masks. For example,
        if an image belongs to the 'skirt' category, the image will have
        up to 4 heatmaps because 'skirt' has at most 4 keypoints.

        Args:
            keypoints (tensor): [num_keypoints, 3]
                Each keypoint contains a (x, y, visibility) tuple.
            input_size (int):
                Image input size.
            stride (int):
                Downsample multiplier.
            hm_alpha (float):
                Alpha compositing for controlling the level of transparency.
            hm_sigma (float):
                Standard deviation for Gaussian function.

        Returns:
            img_heatmaps (tensor):
            img_vis_masks (tensor):
        '''
        num_keypoints = len(keypoints)

        # `hm_size = 128` is the desired heatmap size because the output
        # of CPN is 4x downsampled and we scaled all input images to (512, 512).
        hm_size = input_size // stride
        kpts = keypoints.clone()

        # According to the author,
        #
        # "For the `-1`, it's a little bit tricky here. I'm not very sure about
        # it. I guess the origin of the coordinates given might start from 1
        # instead of 0. But maybe I am wrong. It's not a very big deal anyway."
        #
        # Note that since we downsampled the image by 4, we want to downsample
        # the keypoints coordinates by 4 (i.e. stride=4) too.
        kpts[:, :2] = (kpts[:, :2] - 1.0) / stride

        img_heatmaps = torch.zeros([num_keypoints, hm_size, hm_size])

        # `img_vis_masks` shows which keypoints exist for this image.
        # Check the loss function for its usage.
        img_vis_masks = torch.zeros([num_keypoints])

        for i, kpt in enumerate(kpts):
            # For `visibility`, each digit represents the following:
            #
            # 1  = Visible
            # 0  = Not visible (i.e. occlude)
            # -1 = Does not exist (this kpt belongs to this category
            # but the annotation does not exist for this image)
            x, y, visibility = kpt

            if visibility >= 0:
                img_heatmaps[i] = self.__keypoint_to_heatmap(hm_size, x, y,
                                                             hm_alpha,
                                                             hm_sigma)
            img_vis_masks[i] = visibility

        return img_heatmaps, img_vis_masks

    def decode_np(self, heatmaps, scale, stride, img_center):
        '''Given a heatmap, we pick the largest value of each Gaussian
        prediction and returns the keypoints coordinate on the original image.

        Args:
            heatmap (tensor): [num_keypoints, h, w]
                Gaussian heatmaps.
            scale (float):
                Ratio of img_max_size to max(img_w, img_h).
            stride (int):
                Downsample multiplier.
            img_center (tuple):
                Center (x, y) of the input image.

        Returns:
            Two `np.array` in size `num_keypoints`.
        '''
        num_keypoints, h, w = heatmaps.shape

        # Compute the resized center of (128, 128) heatmap.
        scaled_ctr_x, scaled_ctr_y = np.array(img_center) * scale / stride

        # According to the author using "Gaussian blur on predicted
        # heatmap" gives us better performance (-0.5%).
        for i, hm in enumerate(heatmaps):
            heatmaps[i] = cv2.GaussianBlur(hm, (5, 5), 1)

        # Flatten the heatmaps.
        heatmaps_th = heatmaps.reshape(num_keypoints, -1)

        # Sort along the flatten heatmaps. `sorted_idx` has the same size
        # as `heatmaps_th` except now it contains the index along the heatmap
        # axis in sorted order.
        #
        # `sorted_idx[:, -1]` is a tensor in size `num_keypoints`
        # that contains indexes of the largest value w.r.t. to each keypoint.
        # Thus, `sorted_idx[:, -2]` is the 2nd largest.
        #
        # Note that the largest value = 'peak' of the Gaussian heatmap.
        sorted_idx = np.argsort(heatmaps_th, axis=1)

        # Each of x1, y1, x2, y2 has size equals to `num_keypoints`.
        # Each (x, y) pair is a keypoint in the (128, 128) heatmaps.
        #
        # Check out this link to understand `np.unravel_index`:
        # https://stackoverflow.com/questions/48135736/what-is-an-intuitive-explanation-of-np-unravel-index
        y1, x1 = np.unravel_index(sorted_idx[:, -1], [h, w])
        y2, x2 = np.unravel_index(sorted_idx[:, -2], [h, w])

        # (y1, x1) is the best predicted keypoints while (y2, x2) is the
        # 2nd best predicted keypoints. We weighed (y1, x1) more than (y2, x2),
        # so (x, y) is now a weighted average of best kpts and 2nd best kpts.
        x, y = (3. * x1 + x2) / 4, (3. * y1 + y2) / 4

        # If the heatmap has variance less than 1, it means the keypoint
        # is out of the image (sometimes this happens when we rotated the image).
        # Instead of not picking any points, we picks the center point of the
        # heatmap as a naive prediction.
        var = np.var(heatmaps_th, axis=1)
        x[var < 1], y[var < 1] = scaled_ctr_x, scaled_ctr_y

        # Transform (x, y) pairs back to the original (512, 512) scale.
        x = x * stride / scale
        y = y * stride / scale

        # According to the author,
        # "Use (x+2, y+2) where (x, y) is max value coordinate (-0.4%)"
        return np.rint(x + 2), np.rint(y + 2)

    def __keypoint_to_heatmap(self, size, mu_x, mu_y, alpha, sigma):
        """
        Generate a bivariate (2D) Gaussian heatmap.

        Args:
            size (int):
                Size of the heatmap
            mu_x, mu_y (float):
                Means for Gaussian function.
            alpha (float):
                Alpha compositing for controlling the level of transparency.
            sigma (float):
                Standard deviation for Gaussian function.

        Returns:
            A (size, size) Gaussian heatmap.
        """
        # x and y are each a size 128 vector: [0, 1, ..., 127]
        x = torch.linspace(0, size - 1, steps=size)
        y = x[:, None]

        # Generate the heatmap using the Gaussian function.
        Z = torch.exp(-(((x-mu_x)**2) / (2*sigma**2) + ((y-mu_y)**2) / (2*sigma**2)))
        return alpha * Z
