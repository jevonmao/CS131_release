import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from scipy.spatial.distance import cdist
from scipy.ndimage import convolve
from utils import pad, describe_keypoints, warp_image, get_output_space

def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve,
        which is already imported above. If you use convolve(), remember to
        specify zero-padding to match our equations, for example:

            out_image = convolve(in_image, kernel, mode='constant', cval=0)

        You can also use for nested loops compute M and the subsequent Harris
        corner response for each output pixel, intead of using convolve().
        Your implementation of conv_fast or conv_nested in HW1 may be a
        useful reference!

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    # 1. Compute x and y derivatives (I_x, I_y) of an image
    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    ### YOUR CODE HERE
    Ixx = dx ** 2
    Ixy = dx * dy
    Iyy = dy ** 2

    Sxx = convolve(Ixx, window, mode='constant', cval=0)
    Sxy = convolve(Ixy, window, mode='constant', cval=0)
    Syy = convolve(Iyy, window, mode='constant', cval=0)

    detM = (Sxx * Syy) - (Sxy ** 2)
    traceM = Sxx + Syy
    response = detM - k * (traceM ** 2)
    ### END YOUR CODE

    return response

def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array.

    The normalization will make the descriptor more robust to change
    in lighting condition.

    Hint:
        In this case of normalization, if a denominator is zero, divide by 1 instead.

    Args:
        patch: grayscale image patch of shape (H, W)

    Returns:
        feature: 1D array of shape (H * W)
    """

    feature = []

    ### YOUR CODE HERE
    patch_mean = np.mean(patch)
    patch_std = np.std(patch)
    if patch_std == 0:
        patch_std = 1
    normalized_patch = (patch - patch_mean) / patch_std
    feature = normalized_patch.flatten()
    ### END YOUR CODE

    return feature

def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be STRICTLY SMALLER
    than the threshold (NOT equal to). Return the matches as pairs of vector indices.

    Hint:
        The Numpy functions np.sort, np.argmin, np.asarray might be useful

        The Scipy function cdist calculates Euclidean distance between all
        pairs of inputs
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints

    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair
        of matching descriptors
    """

    matches = []

    M = desc1.shape[0]
    dists = cdist(desc1, desc2)

    ### YOUR CODE HERE
    if dists.shape[1] < 2:
        matches = np.empty((0, 2), dtype=int)
    else:
        sorted_indices = np.argsort(dists, axis=1)
        best_matches = sorted_indices[:, 0]
        second_best_matches = sorted_indices[:, 1]
        best_dists = dists[np.arange(M), best_matches]
        second_best_dists = dists[np.arange(M), second_best_matches]
        valid_mask = (second_best_dists > 0) & ((best_dists / second_best_dists) < threshold)
        matches = np.column_stack((np.where(valid_mask)[0], best_matches[valid_mask]))
    ### END YOUR CODE

    return matches

def fit_affine_matrix(p1, p2, to_pad=True):
    """
    Fit affine matrix such that p2 * H = p1. First, pad the descriptor vectors
    with a 1 using pad() to convert to homogeneous coordinates, then return
    the least squares fit affine matrix in homogeneous coordinates.

    Hint:
        You can use np.linalg.lstsq function to solve the problem.

        Explicitly specify np.linalg.lstsq's new default parameter rcond=None
        to suppress deprecation warnings, and match the autograder.

    Args:
        p1: an array of shape (M, P) holding descriptors of size P about M keypoints
        p2: an array of shape (M, P) holding descriptors of size P about M keypoints

    Return:
        H: a matrix of shape (P+1, P+1) that transforms p2 to p1 in homogeneous
        coordinates
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'

    if to_pad:
        p1 = pad(p1)
        p2 = pad(p2)

    ### YOUR CODE HERE
    H, _, _, _ = np.linalg.lstsq(p2, p1, rcond=None)
    ### END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H

def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation:

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers via Euclidean distance
        4. Keep the largest set of inliers (use >, i.e. break ties by whichever set is seen first)
        5. Re-compute least-squares estimate on all of the inliers

    Update max_inliers as a boolean array where True represents the keypoint
    at this index is an inlier, while False represents that it is not an inlier.

    Hint:
        You can use fit_affine_matrix to compute the affine transformation matrix.
        Make sure to pass in to_pad=False, since we pad the matches for you here.

        You can compute elementwise boolean operations between two numpy arrays,
        and use boolean arrays to select array elements by index:
        https://numpy.org/doc/stable/reference/arrays.indexing.html#boolean-array-indexing

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """

    # Copy matches array, to avoid overwriting it
    orig_matches = matches.copy()
    matches = matches.copy()

    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N, dtype=bool)
    n_inliers = 0

    # RANSAC iteration start

    # Note: while there're many ways to do random sampling, we use
    # `np.random.shuffle()` followed by slicing out the first `n_samples`
    # matches here in order to align with the autograder.
    # Sample with this code:
    for i in range(n_iters):
        # 1. Select random set of matches
        np.random.shuffle(matches)
        samples = matches[:n_samples]
        sample1 = pad(keypoints1[samples[:,0]])
        sample2 = pad(keypoints2[samples[:,1]])

    ### YOUR CODE HERE
        H_candidate = fit_affine_matrix(sample1, sample2, to_pad=False)
        transformed = matched2 @ H_candidate
        dists = np.linalg.norm(matched1[:, :2] - transformed[:, :2], axis=1)
        inliers = dists < threshold

        if np.sum(inliers) > n_inliers:
            n_inliers = np.sum(inliers)
            max_inliers = inliers

    final_inliers_1 = matched1[max_inliers]
    final_inliers_2 = matched2[max_inliers]

    H = fit_affine_matrix(final_inliers_1, final_inliers_2, to_pad=False)

    ### END YOUR CODE

    return H, orig_matches[max_inliers]

def linear_blend(img1_warped, img2_warped):
    """
    Linearly blend img1_warped and img2_warped by following the steps:

    1. Define left and right margins (already done for you)
    2. Define a weight matrices for img1_warped and img2_warped
        np.linspace and np.tile functions will be useful
    3. Apply the weight matrices to their corresponding images
    4. Combine the images

    Args:
        img1_warped: Refernce image warped into output space
        img2_warped: Transformed image warped into output space

    Returns:
        merged: Merged image in output space
    """

    out_H, out_W = img1_warped.shape # Height and width of output space
    img1_mask = (img1_warped != 0)  # Mask == 1 inside the image
    img2_mask = (img2_warped != 0)  # Mask == 1 inside the image

    # Find column of middle row where warped image 1 ends
    # This is where to end weight mask for warped image 1
    right_margin = out_W - np.argmax(np.fliplr(img1_mask)[out_H//2, :].reshape(1, out_W), 1)[0]

    # Find column of middle row where warped image 2 starts
    # This is where to start weight mask for warped image 2
    left_margin = np.argmax(img2_mask[out_H//2, :].reshape(1, out_W), 1)[0]

    ### YOUR CODE HERE
    weight1 = np.zeros(out_W)
    weight2 = np.zeros(out_W)
    weight1[:left_margin] = 1
    if right_margin < out_W:
        weight2[right_margin:] = 1
    if right_margin >= left_margin:
        blend_width = right_margin - left_margin
        ramp = np.linspace(1, 0, blend_width)
        weight1[left_margin:right_margin] = ramp
        weight2[left_margin:right_margin] = 1 - ramp
    else:
        overlap_slice = slice(left_margin, right_margin+1)
        weight1[overlap_slice] = 0.5
        weight2[overlap_slice] = 0.5
    weight1_mat = np.tile(weight1, (out_H, 1))
    weight2_mat = np.tile(weight2, (out_H, 1))
    merged = weight1_mat * img1_warped + weight2_mat * img2_warped
    ### END YOUR CODE

    return merged

def stitch_multiple_images(imgs, desc_func=simple_descriptor, patch_size=5):
    """
    Stitch an ordered chain of images together using custom RANSAC for robust affine estimation.
    
    For images I1, I2, ..., Im we:
      1. Detect keypoints (using a Harris detector and non-maximum suppression).
      2. Compute descriptors (using desc_func on image patches).
      3. Match descriptors between each consecutive pair.
      4. For each pair, use RANSAC to estimate an affine transform H mapping points in I{i+1}
         to I{i}.
      5. Chain the transformations so that each image is mapped into the coordinate frame of I1.
      6. Compute an output canvas large enough for all warped images.
      7. Warp each image and blend them (by averaging in overlapping regions).
    
    Args:
        imgs: List of m images (assumed grayscale).
        desc_func: Function that takes an image patch and returns a 1D descriptor.
        patch_size: Size of the patch around each keypoint.
    
    Returns:
        panorama: The final stitched panorama.
    """
    # 1. Detect keypoints in each image.
    keypoints = []
    for img in imgs:
        # (Assuming you have functions harris_corners and corner_peaks available.)
        kypnts = corner_peaks(harris_corners(img, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
        keypoints.append(kypnts)
    
    # 2. Compute descriptors at each keypoint.
    descriptors = []
    for i, kypnts in enumerate(keypoints):
        desc = describe_keypoints(imgs[i], kypnts,
                                  desc_func=desc_func,
                                  patch_size=patch_size)
        descriptors.append(desc)
    
    # Match keypoints in neighboring images
    matches = []  # matches[i] corresponds to matches between
                  # descriptors[i] and descriptors[i+1]
    for i in range(len(imgs)-1):
        mtchs = match_descriptors(descriptors[i], descriptors[i+1], 0.7)
        matches.append(mtchs)
    
    ### YOUR CODE HERE
    transforms = []
    for i in range(len(imgs) - 1):
        H_matrix, _ = ransac(
            keypoints[i], 
            keypoints[i + 1], 
            matches[i]
        )
        transforms.append(H_matrix)

    accumulated_transforms = [np.eye(3)]
    for H_matrix in transforms:
        new_transform = (
            H_matrix @ accumulated_transforms[-1]
        )
        accumulated_transforms.append(new_transform)

    final_canvas = np.zeros((0, 0))
    max_offset = np.zeros(2)

    for i in range(len(imgs)):
        temp_canvas, temp_offset = get_output_space(
            imgs[i], 
            imgs, 
            accumulated_transforms
        )
        
        if temp_canvas.shape[0] > final_canvas.shape[0]:
            final_canvas = temp_canvas
            max_offset = temp_offset

    aligned_images = []
    for i in range(len(imgs)):
        transformed_img = warp_image(
            imgs[i], 
            accumulated_transforms[i], 
            final_canvas, 
            max_offset
        )
        valid_mask = (transformed_img != -1)
        transformed_img[~valid_mask] = 0
        aligned_images.append(transformed_img)

    stitched_panorama = aligned_images[0]
    for i in range(1, len(aligned_images)):
        stitched_panorama = linear_blend(
            stitched_panorama, 
            aligned_images[i]
        )
    panorama = stitched_panorama
    ### END YOUR CODE
    return panorama


