import numpy as np
import cv2

def low_rank_approximation(matrix, k):
    U, S, V = np.linalg.svd(matrix)
    # print(U.shape, S.shape, V.shape)
    return U[:, :k].dot(np.diag(S[:k])).dot(V[:k, :])
    # return np.dot(U[:,:291] * S, V)


img = cv2.imread("image.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print("Original image:")
print(img)
print(img.shape)
img2 = low_rank_approximation(img, 50).round().astype(type(img[0][0]))
print(img2.shape)
print("Reconstructed image:")
print(img2)
cv2.imshow('Grayscale', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()