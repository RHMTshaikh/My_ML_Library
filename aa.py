import numpy as np
import timeit

class Layer:
    def __init__(self, kernel_size, stride, output_shape, input_shape=None, padded_input_shape=None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.padded_input_shape = padded_input_shape 

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
            writeable=True
        )
        
        col_final = col_view.reshape((out_h * out_w, kernel_h * kernel_w, C_in))
        
        return col_final, (out_h, out_w)

    # --- Original Loop-Based col2im Function ---
    def col2im_loop(self, col_data: np.ndarray) -> np.ndarray:
        num_windows, kernel_elements_flat, C_in = col_data.shape

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        
        H_padded, W_padded, _ = self.padded_input_shape
        output_image = np.zeros(self.padded_input_shape, dtype=col_data.dtype)

        out_h_derived = (H_padded - kernel_h) // stride_h + 1
        out_w_derived = (W_padded - kernel_w) // stride_w + 1

        window_idx = 0
        for h_out in range(out_h_derived):
            for w_out in range(out_w_derived):
                h_start = h_out * stride_h
                w_start = w_out * stride_w
                
                current_patch = col_data[window_idx].reshape(kernel_h, kernel_w, C_in)
                
                output_image[h_start : h_start + kernel_h,
                             w_start : w_start + kernel_w,
                             :] += current_patch
                
                window_idx += 1
        
        return output_image

    # --- Vectorized col2im Function ---
    def col2im_vectorized(self, col_data: np.ndarray) -> np.ndarray:
        num_windows, kernel_elements_flat, C_in = col_data.shape

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        
        H_padded, W_padded, _ = self.padded_input_shape
        output_image = np.zeros(self.padded_input_shape, dtype=col_data.dtype)

        out_h_derived = (H_padded - kernel_h) // stride_h + 1
        out_w_derived = (W_padded - kernel_w) // stride_w + 1

        values_flat = col_data.ravel()

        out_h_idx, out_w_idx = np.indices((out_h_derived, out_w_derived))
        out_h_idx = out_h_idx.ravel()
        out_w_idx = out_w_idx.ravel()

        kernel_h_idx, kernel_w_idx = np.indices((kernel_h, kernel_w))
        kernel_h_idx = kernel_h_idx.ravel()
        kernel_w_idx = kernel_w_idx.ravel()
        
        absolute_h_coords = (out_h_idx[:, None] * stride_h + kernel_h_idx[None, :])
        absolute_w_coords = (out_w_idx[:, None] * stride_w + kernel_w_idx[None, :])

        h_indices_flat = np.repeat(absolute_h_coords.ravel(), C_in)
        w_indices_flat = np.repeat(absolute_w_coords.ravel(), C_in)
        c_indices_flat = np.tile(np.arange(C_in), num_windows * kernel_elements_flat)

        np.add.at(output_image, (h_indices_flat, w_indices_flat, c_indices_flat), values_flat)
        
        return output_image


# --- Benchmark Setup ---
# Large input image to clearly see the performance difference
input_H, input_W, input_C = 256, 256, 3 # A typical image size
kernel_size = (3, 3)
stride = (1, 1) # Full overlap, which makes col2im complex and benefits from vectorization

# Calculate output shape for im2col
out_h_calculated = (input_H - kernel_size[0]) // stride[0] + 1 
out_w_calculated = (input_W - kernel_size[1]) // stride[1] + 1 
output_shape_for_im2col = (out_h_calculated, out_w_calculated, input_C)

# Instantiate the layer
layer = Layer(
    kernel_size=kernel_size, 
    stride=stride,
    output_shape=output_shape_for_im2col,
    input_shape=(input_H, input_W, input_C),
    padded_input_shape=(input_H, input_W, input_C) # Assuming no actual padding for this example
)

# Create a dummy input image and col_data
input_image = np.random.rand(input_H, input_W, input_C)
col_data, _ = layer.im2col(input_image) # Get actual col_data from im2col for realistic test

print(f"Benchmarking col2im for input_image: {input_image.shape}, kernel: {kernel_size}, stride: {stride}")
print(f"col_data shape: {col_data.shape}")

num_runs = 100 # Number of times to run each function for averaging

# Benchmark loop-based col2im
time_loop = timeit.timeit(lambda: layer.col2im_loop(col_data), number=num_runs)
print(f"\nLoop-based col2im: {time_loop / num_runs:.6f} seconds per run")

# Benchmark vectorized col2im
time_vectorized = timeit.timeit(lambda: layer.col2im_vectorized(col_data), number=num_runs)
print(f"Vectorized col2im: {time_vectorized / num_runs:.6f} seconds per run")

speedup = (time_loop / num_runs) / (time_vectorized / num_runs)
print(f"\nVectorized is {speedup:.2f}x faster than loop-based.")

# Optional: Verify correctness (they should produce the same result)
result_loop = layer.col2im_loop(col_data)
result_vectorized = layer.col2im_vectorized(col_data)
print(f"\nResults are close (tolerance 1e-9)? {np.allclose(result_loop, result_vectorized, atol=1e-9)}")