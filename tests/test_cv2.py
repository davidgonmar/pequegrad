import pytest
import cv2
import pequegrad.cv2 as pg_cv2
import numpy as np


def load_test_image():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    return img


def compare_images(img1, img2, tol=1e-5):
    np.testing.assert_allclose(img1, img2, atol=tol, rtol=tol)


def test_imread_imwrite():
    img = load_test_image()
    cv2.imwrite("test_img.jpg", img)

    cv2_img = cv2.imread("test_img.jpg")
    pg_img = pg_cv2.imread("test_img.jpg").numpy()

    compare_images(cv2_img, pg_img)
    pg_cv2.imwrite("test_out.jpg", pg_cv2.numpy_to_tensor(pg_img))


def test_resize():
    img = load_test_image()
    cv2_resized = cv2.resize(img, (50, 50), interpolation=cv2.INTER_NEAREST)
    pg_resized = pg_cv2.resize(
        pg_cv2.numpy_to_tensor(img), (50, 50), pg_cv2.INTER_NEAREST
    ).numpy()
    compare_images(cv2_resized, pg_resized)


def AAtest_cvtColor():
    img = load_test_image()
    cv2_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pg_gray = (
        pg_cv2.cvtColor(pg_cv2.numpy_to_tensor(img), pg_cv2.COLOR_BGR2GRAY)
        .numpy()
        .squeeze()
    )

    assert compare_images(cv2_gray, pg_gray)

    cv2_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pg_rgb = pg_cv2.cvtColor(pg_cv2.numpy_to_tensor(img), pg_cv2.COLOR_BGR2RGB).numpy()

    assert compare_images(cv2_rgb, pg_rgb)


def test_gaussian_blur():
    img = load_test_image().astype(np.float32) / 255.0
    cv2_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=1.0)
    pg_blur = pg_cv2.GaussianBlur(
        pg_cv2.numpy_to_tensor(img), (5, 5), sigmaX=1.0
    ).numpy()

    assert compare_images(cv2_blur, pg_blur, tol=1e-3)


if __name__ == "__main__":
    pytest.main(["-v", "--tb=short"])
