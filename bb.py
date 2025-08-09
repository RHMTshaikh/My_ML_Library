import numpy as np

class Layer:
    def __init__(self, kernel_size, stride, output_shape, input_shape=None, padded_input_shape=None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.padded_input_shape = padded_input_shape # Used as the target shape for col2im

    # Your provided im2col function
    def im2col(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        H, W, C_in = image.shape
        
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        out_h, out_w, _ = self.output_shape 

        strides_H, strides_W, strides_C = image.strides

        view_shape = (
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            C_in,
        )
        new_strides = (
            strides_H * stride_h,
            strides_W * stride_w,
            strides_H,
            strides_W,
            strides_C,
        )
        col_view = np.lib.stride_tricks.as_strided(
            image,
            shape=view_shape,
            strides=new_strides,
            writeable=True # Important if `image` might be modified indirectly
        )
        
        # This reshape likely creates a copy. It flattens the kernel elements.
        col_final = col_view.reshape((out_h * out_w, kernel_h * kernel_w, C_in))
        
        return col_final, (out_h, out_w)

    # --- Vectorized col2im Function ---
    def col2im(self, col_data: np.ndarray) -> np.ndarray:
        # col_data_shape: (NumWindows, KernelElements, C_in)
        num_windows, kernel_elements_flat, C_in = col_data.shape

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        # Ensure padded_input_shape is set and is valid
        if self.padded_input_shape is None:
            raise ValueError("self.padded_input_shape must be set for col2im output shape.")
        H_padded, W_padded, _ = self.padded_input_shape

        # Initialize the output image with zeros. This is where contributions will sum up.
        output_image = np.zeros(self.padded_input_shape, dtype=col_data.dtype)

        # Calculate out_h and out_w based on padded_input_shape and layer parameters
        # This is derived from the formula H_out = (H_in - K_H) / S_H + 1
        # So, H_in = (H_out - 1) * S_H + K_H
        # We need the out_h, out_w from the input (col_data) to ensure consistency.
        # Given col_data's num_windows = out_h * out_w:
        out_h_derived = (H_padded - kernel_h) // stride_h + 1
        out_w_derived = (W_padded - kernel_w) // stride_w + 1

        if num_windows != out_h_derived * out_w_derived:
             raise ValueError(f"Number of windows in col_data ({num_windows}) does not match "
                              f"calculated output dimensions ({out_h_derived}*{out_w_derived}). "
                              "Check input shapes or layer parameters.")

        # --- Vectorized Index Generation ---
        # 1. Generate the starting (top-left) row/col coordinates for each window
        # These are the coordinates in the output_image
        rows_starts = np.arange(out_h_derived) * stride_h
        cols_starts = np.arange(out_w_derived) * stride_w

        # 2. Generate the relative row/col coordinates *within* each kernel
        # These are (0,0) to (kernel_h-1, kernel_w-1)
        kernel_rows_relative = np.arange(kernel_h)
        kernel_cols_relative = np.arange(kernel_w)

        # 3. Create a meshgrid of absolute image coordinates for all elements in all windows
        # Use broadcasting to combine start positions with relative positions
        # Absolute row coordinates for all kernel elements across all windows
        # shape: (out_h, out_w, kernel_h, kernel_w)
        abs_rows = (rows_starts[:, None, None, None] + kernel_rows_relative[None, None, :, None])
        abs_cols = (cols_starts[None, :, None, None] + kernel_cols_relative[None, None, None, :])
        
        # Flatten these coordinates to match the col_data structure for indexing
        # Reshape col_data to (out_h, out_w, kernel_h, kernel_w, C_in) first
        col_data_reshaped = col_data.reshape(out_h_derived, out_w_derived, kernel_h, kernel_w, C_in)

        # Flatten abs_rows and abs_cols to be 1D lists of indices
        # We need to broadcast the channel dimension explicitly for indexing
        # For each element in col_data_reshaped, we need its (h, w, c) index in output_image.
        # abs_rows, abs_cols are 4D, col_data_reshaped is 5D. We need 3D output indices.

        # Correct absolute indexing:
        # Each element in col_data_reshaped[oh, ow, kh, kw, c_in]
        # corresponds to output_image[oh*sh + kh, ow*sw + kw, c_in]

        # Use np.add.at for efficient scatter-add.
        # np.add.at(array, indices, values)
        # Indices need to be tuples of arrays for each dimension.

        # Create all absolute row indices for all elements in col_data
        # This will be (out_h, out_w, kernel_h, kernel_w)
        image_row_indices = (rows_starts[:, None] + kernel_rows_relative).reshape(-1) # Flattened to (out_h * kernel_h)
        image_col_indices = (cols_starts[:, None] + kernel_cols_relative).reshape(-1) # Flattened to (out_w * kernel_w)

        # Correct way to get flattened indices for np.add.at (less intuitive to write for 5D)
        # It's generally easier to iterate over channels, or reform the col_data.
        # The key is to generate the (H, W) indices that correspond to the (NumWindows, KernelElements)
        # for each channel.

        # The most straightforward way to use np.add.at without explicit Python loops:
        # Create an array of (out_h * out_w) x (kernel_h * kernel_w) x C_in coordinates
        # to map back to (H_padded, W_padded, C_in)

        # Generate all (h_out, w_out) pairs
        out_h_idx, out_w_idx = np.indices((out_h_derived, out_w_derived))
        out_h_idx = out_h_idx.ravel() # Flatten to (num_windows,)
        out_w_idx = out_w_idx.ravel() # Flatten to (num_windows,)

        # Generate all (kernel_h, kernel_w) pairs
        kernel_h_idx, kernel_w_idx = np.indices((kernel_h, kernel_w))
        kernel_h_idx = kernel_h_idx.ravel() # Flatten to (kernel_elements_flat,)
        kernel_w_idx = kernel_w_idx.ravel() # Flatten to (kernel_elements_flat,)

        # Compute absolute H and W coordinates for all elements in all windows
        # Use broadcasting:
        # (num_windows, 1) * stride_h + (1, kernel_elements_flat)
        absolute_h_coords = (out_h_idx[:, None] * stride_h + kernel_h_idx[None, :])
        absolute_w_coords = (out_w_idx[:, None] * stride_w + kernel_w_idx[None, :])

        # Repeat for each channel
        # Reshape col_data to (num_windows, kernel_elements_flat, C_in)
        # And the coords to match: (num_windows, kernel_elements_flat, 1) for broadcasting
        
        # Prepare indices for np.add.at
        # These need to be 1D arrays for each dimension of the target `output_image`.
        # Shape of `col_data` is (num_windows, kernel_elements_flat, C_in)
        # We need to flatten col_data to (num_windows * kernel_elements_flat * C_in)
        # And create corresponding flat index arrays for H, W, C.

        # Flatten col_data: (num_windows * kernel_elements_flat * C_in,)
        values_flat = col_data.ravel()

        # Create flattened absolute H indices
        h_indices_flat = (out_h_idx[:, None] * stride_h + kernel_h_idx[None, :]).ravel()
        # Create flattened absolute W indices
        w_indices_flat = (out_w_idx[:, None] * stride_w + kernel_w_idx[None, :]).ravel()
        
        # Now replicate these for each channel
        h_indices_flat = np.repeat(h_indices_flat, C_in)
        w_indices_flat = np.repeat(w_indices_flat, C_in)
        
        # Create flattened C indices (0,1,2,0,1,2,...)
        c_indices_flat = np.tile(np.arange(C_in), num_windows * kernel_elements_flat)

        # Perform the scatter-add operation
        np.add.at(output_image, (h_indices_flat, w_indices_flat, c_indices_flat), values_flat)
        
        return output_image


# --- Example Usage ---
# Dummy input image (H, W, C_in)
input_H, input_W, input_C = 6, 6, 3
input_image = np.arange(input_H * input_W * input_C, dtype=float).reshape(input_H, input_W, input_C)

# Pooling/Convolution parameters
kernel_size = (3, 3)
stride = (2, 2)

# Calculate expected output shape for im2col
out_h_calculated = (input_H - kernel_size[0]) // stride[0] + 1 # (6 - 3) // 2 + 1 = 2
out_w_calculated = (input_W - kernel_size[1]) // stride[1] + 1 # (6 - 3) // 2 + 1 = 2
output_shape_for_im2col = (out_h_calculated, out_w_calculated, input_C)

# Instantiate the layer
layer = Layer(kernel_size=kernel_size, stride=stride,
              output_shape=output_shape_for_im2col, # Used internally by im2col
              input_shape=input_image.shape,        # Input shape of image
              padded_input_shape=input_image.shape) # The shape of the desired output image from col2im

print("Original Image (Input for im2col):")
print(input_image[:,:,0]) # Show first channel
print(f"Shape: {input_image.shape}\n")

# --- Test im2col ---
col_data, (out_h, out_w) = layer.im2col(input_image)
print("col_data (Output of im2col):")
print(f"Shape: {col_data.shape}\n") # Expected (out_h * out_w, kernel_h * kernel_w, C_in) = (4, 9, 3)

# --- Test col2im (vectorized) ---
reconstructed_image = layer.col2im(col_data)

print("Reconstructed Image (Output of col2im - will show summed overlaps):")
print(reconstructed_image[:,:,0]) # Show first channel
print(f"Shape: {reconstructed_image.shape}\n")

# --- Verification (critical for overlapping regions) ---
# If stride < kernel_size, regions overlap, and values are summed.
# The `reconstructed_image` will contain summed values where patches overlapped.
# To confirm correctness, you might want to test with a dummy `col_data`
# (e.g., all ones) and verify the expected sum at overlapping pixels.

# Example: If col_data was all ones, and a pixel is covered by 2 patches, its value should be 2.
# Let's create a col_data with all ones for verification:
ones_col_data = np.ones_like(col_data, dtype=float)
reconstructed_ones = layer.col2im(ones_col_data)
print("Reconstructed Image from all-ones col_data (shows overlap counts):")
print(reconstructed_ones[:,:,0]) # The values show how many patches cover each pixel

# Expected output for reconstructed_ones[:,:,0] with 3x3 kernel, stride 2x2 on 6x6:
# Top-left (0,0) is covered by 1 patch.
# (0,1) covered by 1.
# (0,2) covered by 2 (from first window's top right, and second window's top left)
# (0,3) covered by 1
# (0,4) covered by 1
# (0,5) covered by 1
# (1,0) covered by 1
# (1,1) covered by 1
# (1,2) covered by 2
# (1,3) covered by 1
# (1,4) covered by 1
# (1,5) covered by 1
# (2,0) covered by 2 (from first window's bottom left, and window below's top left)
# (2,1) covered by 2
# (2,2) covered by 4 (from 4 overlapping windows)
# etc.