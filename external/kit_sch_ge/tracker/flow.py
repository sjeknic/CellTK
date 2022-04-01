"""Estimate shift between image crops using phase correlation."""
import numpy as np
from scipy.fftpack import fftn, ifftn


def compute_fft_displacement(img1, img2):
    """
    Estimates shift between images via phase correlation.
    Args:
        img1: np.array
        img2: np.array

    Returns: a vector containing the estimated displacement between the two image crops

    """
    img_shape = np.array(img1.shape)
    img_filter = [np.hanning(s) for s in img_shape]
    if len(img1.shape) == 2:
        img_filter = img_filter[0].reshape(-1, 1) * img_filter[1].reshape(1, -1)
    elif len(img1.shape) == 3:
        img_filter = img_filter[0].reshape(-1, 1, 1) * img_filter[1].reshape(1, -1, 1) * img_filter[2].reshape(1, 1, -1)

    fft1 = fftn(img1 * img_filter)
    fft2 = fftn(img2 * img_filter)
    quotient = np.conj(fft1) * fft2 / (np.abs(np.conj(fft1) * fft2)+1e-12)  # elementwise multiplication !
    correlation = ifftn(quotient)
    # estimate tau:=t_2 - t_1
    peak = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)
    peak = np.array(peak)
    # peak between 0...shape positive shift: displacement==shift,
    # negative shift: displacement=shape-shift due to circularity (fft)
    negative_shift = peak > (img_shape // 2)
    displacement = peak
    displacement[negative_shift] = -(img_shape - peak)[negative_shift]
    return displacement


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage.morphology import disk, ball

    DISK_SIZE = 10
    POS_1_2D = (30, 30)
    POS_2_2D = (15, 15)

    BALL_SIZE = 10
    POS_1_3D = (55, 30, 30)
    POS_2_3D = (45, 15, 10)

    IMG_1_2D = np.zeros((100, 100))
    IMG_2_2D = np.zeros((100, 100))

    DISK_2D = disk(DISK_SIZE)
    IMG_1_2D[POS_1_2D[0] - DISK_SIZE:POS_1_2D[0] + DISK_SIZE + 1,
             POS_1_2D[1] - DISK_SIZE:POS_1_2D[1] + DISK_SIZE + 1] = DISK_2D
    IMG_2_2D[POS_2_2D[0] - DISK_SIZE:POS_2_2D[0] + DISK_SIZE + 1,
             POS_2_2D[1] - DISK_SIZE:POS_2_2D[1] + DISK_SIZE + 1] = DISK_2D
    plt.imshow(IMG_1_2D + IMG_2_2D)
    plt.show()
    print(compute_fft_displacement(IMG_1_2D, IMG_2_2D))

    # 3D
    IMG_1_3D = np.zeros((100, 100, 50))
    IMG_2_3D = np.zeros((100, 100, 50))

    BALL_3D = ball(BALL_SIZE)

    IMG_1_3D[POS_1_3D[0] - BALL_SIZE:POS_1_3D[0] + BALL_SIZE + 1,
             POS_1_3D[1] - BALL_SIZE:POS_1_3D[1] + BALL_SIZE + 1,
             POS_1_3D[2] - BALL_SIZE:POS_1_3D[2] + BALL_SIZE + 1] = BALL_3D

    IMG_2_3D[POS_2_3D[0] - BALL_SIZE:POS_2_3D[0] + BALL_SIZE + 1,
             POS_2_3D[1] - BALL_SIZE:POS_2_3D[1] + BALL_SIZE + 1,
             POS_2_3D[2] - BALL_SIZE:POS_2_3D[2] + BALL_SIZE + 1] = BALL_3D
    _, ax = plt.subplots(1, 3)
    ax[0].imshow(np.max(IMG_1_3D + IMG_2_3D, axis=0))
    ax[1].imshow(np.max(IMG_1_3D + IMG_2_3D, axis=1))
    ax[2].imshow(np.max(IMG_1_3D + IMG_2_3D, axis=2))
    plt.show()
    print(compute_fft_displacement(IMG_1_3D, IMG_2_3D))
