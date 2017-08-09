
from theano import tensor as T





def mirror_padding(images):
    """
    Mirror padding is used to apply a 2D convolution avoiding the border
    effects that one normally gets with zero padding.
    We assume that the filter has an odd size.
    To obtain a filtered tensor with the same output size, substitute
    a ``conv2d(images, filters, mode="half")`` with
    ``conv2d(mirror_padding(images, filters.shape), filters, mode="valid")``.
    Parameters
    ----------
    images : Tensor
        4D tensor containing a set of images.
    filter_size : tuple
        Spatial size of the filter (height, width).
    Returns
    -------
    padded : Tensor
        4D tensor containing the padded set of images.
    """
    padding = ((1, 1), (1, 1))
    top_pad, bottom_pad = padding[0]
    left_pad, right_pad = padding[1]
    s = images.shape
    padded_shape = (s[0],
                   s[1],
                   s[2] + top_pad + bottom_pad,
                   s[3] + left_pad + right_pad)
    x_padded = T.zeros(padded_shape)

    # h_pad = padding[0]
    # w_pad = padding[1]
    #
    # # Allocate space for padded images.
    # s = images.shape
    # padded_shape = (s[0], s[1], s[2] + 2*h_pad, s[3] + 2*w_pad)
    #
    # x_padded = T.zeros(padded_shape)

    # Copy the original image to the central part.
    x_padded = T.set_subtensor(
        x_padded[:, :, top_pad:s[2]+bottom_pad, left_pad:right_pad+s[3]],
        images,
    )

    # Copy borders.
    # Note that we don't mirror starting at pixel number 0: assuming that
    # we have a symmetric, odd filter, the central element of the filter
    # will run along the original border, and we need to match the
    # statistics of the elements around it.
    x_padded = T.set_subtensor(
        x_padded[:, :, :top_pad, left_pad:-right_pad],
        images[:, :, top_pad:0:-1, :],
    )
    x_padded = T.set_subtensor(
        x_padded[:, :, -bottom_pad:, left_pad:-right_pad],
        images[:, :, -2:-bottom_pad-2:-1, :],
    )
    x_padded = T.set_subtensor(
        x_padded[:, :, :, :right_pad],
        x_padded[:, :, :, 2*left_pad:right_pad:-1],
    )
    x_padded = T.set_subtensor(
        x_padded[:, :, :, -left_pad:],
        x_padded[:, :, :, -left_pad-2:-2*right_pad-2:-1],
    )

    return x_padded


def mean_padding(images):
    """
    Mirror padding is used to apply a 2D convolution avoiding the border
    effects that one normally gets with zero padding.
    We assume that the filter has an odd size.
    To obtain a filtered tensor with the same output size, substitute
    a ``conv2d(images, filters, mode="half")`` with
    ``conv2d(mirror_padding(images, filters.shape), filters, mode="valid")``.
    Parameters
    ----------
    images : Tensor
        4D tensor containing a set of images.
    filter_size : tuple
        Spatial size of the filter (height, width).
    Returns
    -------
    padded : Tensor
        4D tensor containing the padded set of images.
    """
    padding = ((1, 1), (1, 1))
    top_pad, bottom_pad = padding[0]
    left_pad, right_pad = padding[1]
    s = images.shape
    padded_shape = (s[0],
                   s[1],
                   s[2] + top_pad + bottom_pad,
                   s[3] + left_pad + right_pad)
    x_padded = T.zeros(padded_shape)


    x_padded = T.set_subtensor(
        x_padded[:, :, top_pad:s[2]+bottom_pad, left_pad:right_pad+s[3]],
        images,
    )

    # Copy borders.
    # Note that we don't mirror starting at pixel number 0: assuming that
    # we have a symmetric, odd filter, the central element of the filter
    # will run along the original border, and we need to match the
    # statistics of the elements around it.
    x_padded = T.set_subtensor(
        x_padded[:, :, :top_pad, left_pad:-right_pad],
        images[:, :, top_pad:0:-1, :],
    )
    x_padded = T.set_subtensor(
        x_padded[:, :, -bottom_pad:, left_pad:-right_pad],
        images[:, :, -2:-bottom_pad-2:-1, :],
    )
    x_padded = T.set_subtensor(
        x_padded[:, :, :, :right_pad],
        x_padded[:, :, :, 2*left_pad:right_pad:-1],
    )
    x_padded = T.set_subtensor(
        x_padded[:, :, :, -left_pad:],
        x_padded[:, :, :, -left_pad-2:-2*right_pad-2:-1],
    )

    return x_padded


def padding_shape(input_shape):
    """
    Mirror padding is used to apply a 2D convolution avoiding the border
    effects that one normally gets with zero padding.
    We assume that the filter has an odd size.
    To obtain a filtered tensor with the same output size, substitute
    a ``conv2d(images, filters, mode="half")`` with
    ``conv2d(mirror_padding(images, filters.shape), filters, mode="valid")``.
    Parameters
    ----------
    images : Tensor
        4D tensor containing a set of images.
    filter_size : tuple
        Spatial size of the filter (height, width).
    Returns
    -------
    padded : Tensor
        4D tensor containing the padded set of images.
    """
    padding = ((1, 1), (1, 1))
    top_pad, bottom_pad = padding[0]
    left_pad, right_pad = padding[1]
    s = input_shape
    padded_shape = (s[0],
                   s[1],
                   s[2] + top_pad + bottom_pad,
                   s[3] + left_pad + right_pad)



    return tuple(padded_shape)

def mirror_padding2(images):
    """
    Mirror padding is used to apply a 2D convolution avoiding the border
    effects that one normally gets with zero padding.
    We assume that the filter has an odd size.
    To obtain a filtered tensor with the same output size, substitute
    a ``conv2d(images, filters, mode="half")`` with
    ``conv2d(mirror_padding(images, filters.shape), filters, mode="valid")``.
    Parameters
    ----------
    images : Tensor
        4D tensor containing a set of images.
    filter_size : tuple
        Spatial size of the filter (height, width).
    Returns
    -------
    padded : Tensor
        4D tensor containing the padded set of images.
    """
    padding = ((2, 2), (2, 2))
    top_pad, bottom_pad = padding[0]
    left_pad, right_pad = padding[1]
    s = images.shape
    padded_shape = (s[0],
                   s[1],
                   s[2] + top_pad + bottom_pad,
                   s[3] + left_pad + right_pad)
    x_padded = T.zeros(padded_shape)

    # h_pad = padding[0]
    # w_pad = padding[1]
    #
    # # Allocate space for padded images.
    # s = images.shape
    # padded_shape = (s[0], s[1], s[2] + 2*h_pad, s[3] + 2*w_pad)
    #
    # x_padded = T.zeros(padded_shape)

    # Copy the original image to the central part.
    x_padded = T.set_subtensor(
        x_padded[:, :, top_pad:s[2]+bottom_pad, left_pad:right_pad+s[3]],
        images,
    )

    # Copy borders.
    # Note that we don't mirror starting at pixel number 0: assuming that
    # we have a symmetric, odd filter, the central element of the filter
    # will run along the original border, and we need to match the
    # statistics of the elements around it.
    x_padded = T.set_subtensor(
        x_padded[:, :, :top_pad, left_pad:-right_pad],
        images[:, :, top_pad:0:-1, :],
    )
    x_padded = T.set_subtensor(
        x_padded[:, :, -bottom_pad:, left_pad:-right_pad],
        images[:, :, -2:-bottom_pad-2:-1, :],
    )
    x_padded = T.set_subtensor(
        x_padded[:, :, :, :right_pad],
        x_padded[:, :, :, 2*left_pad:right_pad:-1],
    )
    x_padded = T.set_subtensor(
        x_padded[:, :, :, -left_pad:],
        x_padded[:, :, :, -left_pad-2:-2*right_pad-2:-1],
    )

    return x_padded


def mean_padding2(images):
    """
    Mirror padding is used to apply a 2D convolution avoiding the border
    effects that one normally gets with zero padding.
    We assume that the filter has an odd size.
    To obtain a filtered tensor with the same output size, substitute
    a ``conv2d(images, filters, mode="half")`` with
    ``conv2d(mirror_padding(images, filters.shape), filters, mode="valid")``.
    Parameters
    ----------
    images : Tensor
        4D tensor containing a set of images.
    filter_size : tuple
        Spatial size of the filter (height, width).
    Returns
    -------
    padded : Tensor
        4D tensor containing the padded set of images.
    """
    padding = ((2, 2), (2, 2))
    top_pad, bottom_pad = padding[0]
    left_pad, right_pad = padding[1]
    s = images.shape
    padded_shape = (s[0],
                   s[1],
                   s[2] + top_pad + bottom_pad,
                   s[3] + left_pad + right_pad)
    x_padded = T.zeros(padded_shape) + T.mean(img)


    x_padded = T.set_subtensor(
        x_padded[:, :, top_pad:s[2]+bottom_pad, left_pad:right_pad+s[3]],
        images,
    )

    # Copy borders.
    # Note that we don't mirror starting at pixel number 0: assuming that
    # we have a symmetric, odd filter, the central element of the filter
    # will run along the original border, and we need to match the
    # statistics of the elements around it.
    x_padded = T.set_subtensor(
        x_padded[:, :, :top_pad, left_pad:-right_pad],
        images[:, :, top_pad:0:-1, :],
    )
    x_padded = T.set_subtensor(
        x_padded[:, :, -bottom_pad:, left_pad:-right_pad],
        images[:, :, -2:-bottom_pad-2:-1, :],
    )
    x_padded = T.set_subtensor(
        x_padded[:, :, :, :right_pad],
        x_padded[:, :, :, 2*left_pad:right_pad:-1],
    )
    x_padded = T.set_subtensor(
        x_padded[:, :, :, -left_pad:],
        x_padded[:, :, :, -left_pad-2:-2*right_pad-2:-1],
    )

    return x_padded


def padding_shape2(input_shape):
    """
    Mirror padding is used to apply a 2D convolution avoiding the border
    effects that one normally gets with zero padding.
    We assume that the filter has an odd size.
    To obtain a filtered tensor with the same output size, substitute
    a ``conv2d(images, filters, mode="half")`` with
    ``conv2d(mirror_padding(images, filters.shape), filters, mode="valid")``.
    Parameters
    ----------
    images : Tensor
        4D tensor containing a set of images.
    filter_size : tuple
        Spatial size of the filter (height, width).
    Returns
    -------
    padded : Tensor
        4D tensor containing the padded set of images.
    """
    padding = ((2, 2), (2, 2))
    top_pad, bottom_pad = padding[0]
    left_pad, right_pad = padding[1]
    s = input_shape
    padded_shape = (s[0],
                   s[1],
                   s[2] + top_pad + bottom_pad,
                   s[3] + left_pad + right_pad)



    return tuple(padded_shape)




