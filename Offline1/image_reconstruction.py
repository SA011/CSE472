import numpy as np
import matplotlib.pyplot as plt
import cv2

def low_rank_approximation(matrix, k):
    U, S, V = np.linalg.svd(matrix)
    # print(U.shape, S.shape, V.shape)
    return U[:, :k].dot(np.diag(S[:k])).dot(V[:k, :])
    # return np.dot(U[:,:291] * S, V)


img = cv2.imread("image.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print("Original image:")
# print(img)
# print("Image Shape: ", img.shape)
m = min(img.shape)
# print(m)
step = m ** (1 / 11)
k = 1
R, C = 3, 4
fig, axes = plt.subplots(R, C, figsize=(15,10))
r ,c = 0, 0
while(int(k) <= m):
    new_img = low_rank_approximation(img, round(k)).round().astype(np.uint8)
    # cv2.imshow(f'K = {round(k)}', new_img)
    # print(k)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    axes[r][c].imshow(new_img, cmap='gray')
    axes[r][c].title.set_text(f'K = {round(k)}')
    axes[r][c].title.set_fontsize(10)
    k *= step
    c += 1
    if c >= C:
        r += 1
        c = 0
plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,wspace=0.1, hspace=0.3)
plt.savefig('low_rank_approximation')
plt.show()
# print(k)