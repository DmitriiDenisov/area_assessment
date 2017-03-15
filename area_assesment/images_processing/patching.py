from sklearn.feature_extraction.image import extract_patches_2d


def array2patches_sklearn(arr, patch_size=(64, 64)):
    return extract_patches_2d(arr, patch_size)


def array2patches2(arr, patch_size=(64, 64), step_size=32):
    return [arr[i: i + patch_size[0], j: j + patch_size[1]]
            for i in range(0, arr.shape[0], step_size) for j in range(0, arr.shape[1], step_size)]
