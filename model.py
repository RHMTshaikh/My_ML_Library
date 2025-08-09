import sys
import traceback
import struct
from array import array
from os.path  import join

from abc import ABC, abstractmethod
import sys
import traceback
from typing import Literal, Callable
import numpy as np
from PIL import Image

def same_owner(arr1, arr2):
    while arr1.base is not None:
        arr1 = arr1.base
        
    while arr2.base is not None:
        arr2 = arr2.base

    print(arr1 is arr2)
    return arr1 is arr2

class Activation_Functions:

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        s = Activation_Functions.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        t = Activation_Functions.tanh(x)
        return 1 - t**2

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, x * alpha)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1.0, alpha)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def softmax_derivative_1d(x) -> np.ndarray:
        y = Activation_Functions.softmax(x)
        return y * (1 - y)
        
    @staticmethod
    def softmax_derivative_jacobian(x) -> np.ndarray:
        K = x.shape[0]
        s = Activation_Functions.softmax(x) # s will be (K,)
        
        diag_s = np.diag(s) # Shape (K, K)

        s_col_vec = s.reshape(K, 1) # Shape (K, 1)
        outer_s = np.dot(s_col_vec, s_col_vec.T) # Shape (K, K)

        jacobian = diag_s - outer_s
        return jacobian

    @staticmethod
    def identity(x:np.ndarray)->np.ndarray:
        return x
    
    @staticmethod
    def identity_derivative(x: np.ndarray)-> np.ndarray:
        return np.ones_like(x)
    
    # --- Lookup Mechanism ---
    _registry = {
        'relu': relu,
        'sigmoid': sigmoid,
        'tanh': tanh,
        'leaky_relu': leaky_relu,
        'softmax': softmax,
        'indentity': identity,
    }

    _derivative_registry = {
        'relu': relu_derivative,
        'sigmoid': sigmoid_derivative,
        'tanh': tanh_derivative,
        'leaky_relu': leaky_relu_derivative,
        'softmax': softmax_derivative_jacobian,
        'indentity': identity_derivative,
    }

    @classmethod
    def get(cls, name: Literal['relu', 'sigmoid', 'tanh', 'leaky_relu', 'softmax', 'indentity'])-> Callable[[np.ndarray], np.ndarray]:
        """
        Retrieves an activation function by its string name.

        Args:
            name (str): The name of the activation function (e.g., 'relu', 'sigmoid').

        Returns:
            function: The corresponding activation function.

        Raises:
            Exception: If the requested activation function is not found.
        """
        name_lower = name.lower() # Make it case-insensitive
        if name_lower not in cls._registry:
            raise Exception(f"Activation function '{name}' not found. "f"Available functions: {list(cls._registry.keys())}")

        return cls._registry[name_lower]

    @classmethod
    def get_derivative(cls, name: Literal['relu', 'sigmoid', 'tanh', 'leaky_relu', 'softmax', 'indentity'])-> Callable[[np.ndarray], np.ndarray]:
        """
        Retrieves the derivative function for an activation function by its string name.

        Args:
            name (str): The name of the activation function (e.g., 'relu', 'sigmoid').

        Returns:
            function: The corresponding derivative function.

        Raises:
            Exception: If the requested derivative function is not found.
        """
        name_lower = name.lower()
        if name_lower not in cls._derivative_registry:
            raise Exception(f"Derivative for '{name}' not found or not implemented. "f"Available derivatives: {list(cls._derivative_registry.keys())}")
            
        return cls._derivative_registry[name_lower]

class Loss_Functions:
    """
    A class to hold various loss functions and their derivatives,
    and provide a lookup mechanism.
    """

    @staticmethod
    def mse(y_true, y_pred):
        """
        Mean Squared Error (MSE) loss.
        y_true: True labels (e.g., shape (batch_size, num_outputs))
        y_pred: Predicted values (e.g., shape (batch_size, num_outputs))
        """
        return np.mean((y_pred - y_true)**2)

    @staticmethod
    def mse_derivative(y_true, y_pred):
        """
        Derivative of MSE loss with respect to y_pred.
        """
        return 2 * (y_pred - y_true) / y_true.shape[0] # Divide by batch size for mean

    @staticmethod
    def cross_entropy(y_true, y_pred_softmax, epsilon=1e-12) -> int:
        """
        Categorical Cross-Entropy loss.
        Suitable for multi-class classification where y_pred_softmax is
        the output of a Softmax activation.
        y_true: One-hot encoded true labels (shape (batch_size, num_classes))
        y_pred_softmax: Predicted probabilities from Softmax (shape (num_classes, ))
        epsilon: Small value to prevent log(0)
        """
        # Clip predictions to avoid log(0) or log(1) issues
        y_pred_softmax = np.clip(y_pred_softmax, epsilon, 1. - epsilon)
        loss = -np.sum(y_true * np.log(y_pred_softmax)) 
        return loss

    @staticmethod
    def cross_entropy_derivative(y_true, y_pred_softmax) -> np.ndarray:
        """
        Derivative of Cross-Entropy loss with respect to the pre-softmax activations (Z).
        This derivative is combined with the softmax derivative for a very simple form:
        dJ/dZ = y_pred_softmax - y_true.
        """
        # This derivative is typically calculated for the pre-softmax activations (Z)
        # when combined with softmax. The chain rule simplifies:
        # dJ/dZ = dJ/dA * dA/dZ = (y_pred_softmax - y_true) for softmax + cross-entropy
        return (y_pred_softmax - y_true)

    @staticmethod
    def binary_cross_entropy(y_true, y_pred_sigmoid, epsilon=1e-12):
        """
        Binary Cross-Entropy loss.
        Suitable for binary classification where y_pred_sigmoid is
        the output of a Sigmoid activation.
        y_true: True labels (shape (batch_size, 1) or (batch_size,))
        y_pred_sigmoid: Predicted probabilities from Sigmoid (shape (batch_size, 1) or (batch_size,))
        epsilon: Small value to prevent log(0)
        """
        y_pred_sigmoid = np.clip(y_pred_sigmoid, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred_sigmoid) + (1 - y_true) * np.log(1 - y_pred_sigmoid))
        return loss

    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred_sigmoid):
        """
        Derivative of Binary Cross-Entropy loss with respect to the pre-sigmoid activations (Z).
        This derivative is combined with the sigmoid derivative for a very simple form:
        dJ/dZ = y_pred_sigmoid - y_true.
        """
        # Similar to categorical cross-entropy, the chain rule simplifies for sigmoid + BCE:
        # dJ/dZ = dJ/dA * dA/dZ = (y_pred_sigmoid - y_true)
        return (y_pred_sigmoid - y_true) / y_true.shape[0] # Divide by batch size for mean

    # --- Lookup Mechanism ---
    _registry = {
        'mse': mse,
        'cross_entropy': cross_entropy,
        'binary_cross_entropy': binary_cross_entropy,
    }

    _derivative_registry = {
        'mse': mse_derivative,
        'cross_entropy': cross_entropy_derivative,
        'binary_cross_entropy': binary_cross_entropy_derivative,
    }

    @classmethod
    def get_loss_func(cls, name: str):
        """
        Retrieves a loss function by its string name.
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise Exception(f"Loss function '{name}' not found. "
                            f"Available functions: {list(cls._registry.keys())}")
        return cls._registry[name_lower]

    @classmethod
    def get_derivative_func(cls, name: str):
        """
        Retrieves the derivative function for a loss by its string name.
        """
        name_lower = name.lower()
        if name_lower not in cls._derivative_registry:
            raise Exception(f"Derivative for loss function '{name}' not found. "
                            f"Available derivatives: {list(cls._derivative_registry.keys())}")
        return cls._derivative_registry[name_lower]

class Layer(ABC):
    """
    Abstract base class for all neural network layers.
    This class cannot be instantiated directly.
    Subclasses must implement abstract methods.
    """
    
    type = 'Generic_Layer'
    
    def __init__(self, input_shape: tuple[int, int, int]):
        """
        Args:
            inputs_shape (tuple): The expected shape of the input to this layer.
                                (e.g., (784,) for flattened images, or (28, 28, 1) for Conv2D).
        """
        self.input_shape: tuple[int, int, int] = input_shape
        self.output_shape: tuple[int, ...] | None = None
        
        self.input: np.ndarray | None = None  
        self.output: np.ndarray | None = None  

    def __repr__(self):
        # A helpful representation for debugging
        return f"{self.__class__.__name__}(input_shape={self.input_shape}, output_shape={self.output_shape})"

    def validate_input_shape(self):
        if self.input.shape != self.input_shape:
            raise Exception (f"Expected input shape {self.input_shape} but recieved {self.input.shape}")

    @abstractmethod
    def forward(self, input: np.ndarray):
        pass
    
    @abstractmethod
    def backpropagation(self, derivatives: np.ndarray, learning_rate: float = None) -> np.ndarray:
        pass

class Convolutional_Layer(Layer):
    type = 'conv'
    
    def __init__(self, 
            input_shape: tuple[int, int, int],
            kernel_size: tuple[int, int],
            padding: tuple[int, int],
            stride: tuple[int, int],
        ):
        
        super().__init__(input_shape)
        self.kernel_size: tuple[int, int] = kernel_size
        self.padding: tuple[int, int] = padding
        self.stride: tuple[int, int] = stride
        self.padded_input = None

        self.output_shape = self.compute_output_feature_shape()
        
        
    def compute_output_feature_shape(self) -> tuple[int, int, int]:
        i_h, i_w, c = self.input_shape
        p_h, p_w = self.padding
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride

        self.h_adjust = (i_h + 2 * p_h - k_h) % s_h
        if self.h_adjust != 0:
            self.h_adjust = s_h - self.h_adjust

        self.w_adjust = (i_w + 2 * p_w - k_w) % s_w
        if self.w_adjust != 0:
            self.w_adjust = s_w - self.w_adjust
        
        self.padded_input_shape = (i_h + 2*p_h + self.h_adjust, i_w + 2*p_w + self.w_adjust, c)

        o_h = (i_h + 2*p_h + self.h_adjust - k_h) // s_h + 1
        o_w = (i_w + 2*p_w + self.w_adjust - k_w) // s_w + 1
        
        return (o_h, o_w)
    
    def get_padded_image(self, image: np.ndarray ):
        padding_h, padding_w = self.padding
        
        padded_image = np.pad(
            image,
            (
                (padding_h, padding_h + self.h_adjust),   # Padding for height
                (padding_w, padding_w + self.w_adjust),   # Padding for width
                (0, 0),                   # No padding for channel dimension
            ),
            mode='constant' # Fills padding with zeros
        )
        
        return padded_image
    
class Filter_Layer(Convolutional_Layer):
    type = 'filter'
    
    def __init__(self, 
            input_shape: tuple[int, int, int],
            kernel_size: tuple[int, int],
            padding: tuple[int, int],
            stride: tuple[int, int],
            num_filters: int,
            activation: str, 
        ):
        
        super().__init__(
            input_shape=input_shape,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.activation_function = Activation_Functions.get(activation)
        self.derivative_function = Activation_Functions.get_derivative(activation)
        
        self.output_shape = (*self.output_shape, num_filters)

        filter_shape = (num_filters, *kernel_size, self.input_shape[-1])
        biases_shape = (num_filters,)
        
        self.filters = np.random.randn(*filter_shape) * 0.01 
        self.biases = np.random.randn(*biases_shape) * 0.01

    def im2col(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        filter_h, filter_w = self.kernel_size
        stride_h, stride_w = self.stride
        H, W, C = image.shape # Dimensions of the image (or padded image)
        out_h, out_w, _ = self.output_shape # Spatial dimensions of the output feature maps
        strides_H, strides_W, strides_C = image.strides

        view_shape = (
            out_h,
            out_w,
            filter_h,
            filter_w,
            C
        )
        new_strides = (
            strides_H * stride_h,  # Stride for output height
            strides_W * stride_w,  # Stride for output width
            strides_H,             # Stride within kernel height
            strides_W,             # Stride within kernel width
            strides_C              # Stride within channels
        )
        
        col_view = np.lib.stride_tricks.as_strided(
            image,
            shape=view_shape,
            strides=new_strides
        )
        
        col_final = col_view.reshape(out_h * out_w, -1)
        
        return col_final, (out_h, out_w)

    def col2im_filter(self, col_data: np.ndarray) -> np.ndarray:
        num_windows, flattened_kernel_elements_and_channels = col_data.shape

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        C_in = self.input_shape[-1] # Original input channels
        
        output_image = np.zeros(self.padded_input_shape, dtype=col_data.dtype)

        out_h, out_w, C = self.output_shape
        
        grad_values_to_scatter = col_data.ravel()

        oh_coords, ow_coords = np.indices((out_h, out_w))
        flat_oh_coords = oh_coords.ravel()
        flat_ow_coords = ow_coords.ravel()

        kh_coords, kw_coords = np.indices((kernel_h, kernel_w))
        flat_kh_coords = kh_coords.ravel() # Flattened kernel height indices
        flat_kw_coords = kw_coords.ravel() # Flattened kernel width indices

        abs_h_coords_per_window_element = (flat_oh_coords[:, None] * stride_h + flat_kh_coords[None, :])
        abs_w_coords_per_window_element = (flat_ow_coords[:, None] * stride_w + flat_kw_coords[None, :])

        h_indices_flat = np.repeat(abs_h_coords_per_window_element.ravel(), C_in)
        w_indices_flat = np.repeat(abs_w_coords_per_window_element.ravel(), C_in)
        
        c_indices_flat = np.tile(np.arange(C_in), num_windows * (kernel_h * kernel_w))

        np.add.at(
            output_image, 
            (
                h_indices_flat,
                w_indices_flat,
                c_indices_flat
            ), 
            grad_values_to_scatter
        )
        
        return output_image


    def apply_filters(self, input: np.ndarray) -> np.ndarray:
        N, K_h, K_w, C = self.filters.shape # N: num_filters, K_h,K_w: kernel_size, C: input_channels

        col_data, (out_h, out_w) = self.im2col(input)
        
        filters_flat = self.filters.reshape(N, -1) # Filters flattened: (num_filters, K_h*K_w*C)
        
        output_flat = col_data @ filters_flat.T

        output_flat += self.biases
        
        return output_flat.reshape(out_h, out_w, N)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data # Store original input for backprop
        self.validate_input_shape()
        self.padded_input = self.get_padded_image(input_data) # Get padded version of input
        
        self.filter_output = self.apply_filters(self.padded_input) # This is Z (pre-activation)
        
        self.output = self.activation_function(self.filter_output) # This is A (post-activation)
        return self.output
    
    def backpropagation(self, derivatives: np.ndarray, learning_rate: float = None) -> np.ndarray:
        l_a = derivatives # Incoming gradients, dL/dA (shape: out_h, out_w, num_filters)
        
        a_z = self.derivative_function(self.filter_output) # dA/dZ
        l_z = l_a * a_z # dL/dA * dA/dZ = dL/dZ (element-wise product)

        l_b = np.sum(l_z, axis=(0,1)) # Summing gradients over spatial dimensions for each filter
        
        num_filters = self.filters.shape[0]
        l_z_col = l_z.reshape(-1, num_filters) 
        
        col_data, _ = self.im2col(self.padded_input)

        l_w = l_z_col.T @ col_data
        l_w = l_w.reshape(self.filters.shape)

        l_col_data = l_z_col @ self.filters.reshape(num_filters, -1)
        
        dx_padded_image = self.col2im_filter(l_col_data)

        padding_h, padding_w = self.padding
        H, W, _ = self.input_shape
        
        dx_input = dx_padded_image[padding_h : padding_h + H,
                                   padding_w : padding_w + W,
                                   :]
        
        l_x = dx_input # This is dL/dX (gradient with respect to the original input)
        
        self.filters -= learning_rate * l_w
        self.biases -= learning_rate * l_b

        return l_x

class Pooling_Layer(Convolutional_Layer):
    type = 'pool'
    
    def __init__(self, 
            input_shape:  tuple[int, int, int],
            kernel_size: tuple[int, int],
            padding: tuple[int, int],
            stride: tuple[int, int],
        ):
        
        super().__init__(
            input_shape=input_shape,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        num_feature_maps = self.input_shape[-1]
        self.output_shape = (*self.output_shape, num_feature_maps)

    def im2col(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        H, W, C_in = image.shape
        
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        out_h, out_w, C = self.output_shape

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
        col_view = np.lib.stride_tricks.as_strided (
            image,
            shape=view_shape,
            strides=new_strides
        )
        
        col_final = col_view.reshape((out_h*out_w, kernel_h*kernel_w, C_in))
        
        return col_final, (out_h, out_w)
    
    def col2im(self, col_data: np.ndarray) -> np.ndarray:
        num_windows, kernel_elements_flat, C_in = col_data.shape

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        
        output_image = np.zeros(self.padded_input_shape, dtype=col_data.dtype)

        out_h, out_w, C = self.output_shape

        values_flat = col_data.ravel()

        out_h_idx, out_w_idx = np.indices((out_h, out_w))
        out_h_idx = out_h_idx.ravel()
        out_w_idx = out_w_idx.ravel()

        kernel_h_idx, kernel_w_idx = np.indices((kernel_h, kernel_w))
        kernel_h_idx = kernel_h_idx.ravel()
        kernel_w_idx = kernel_w_idx.ravel()
        
        absolute_h_coords = (out_h_idx[:, None] * stride_h + kernel_h_idx[None, :])
        absolute_w_coords = (out_w_idx[:, None] * stride_w + kernel_w_idx[None, :])

        h_indices_flat = np.repeat(absolute_h_coords.ravel(), C_in)
        w_indices_flat = np.repeat(absolute_w_coords.ravel(), C_in)
        c_indices_flat = np.tile(np.arange(C_in), num_windows * kernel_h * kernel_w)

        np.add.at(
            output_image, 
            (
                h_indices_flat, 
                w_indices_flat, 
                c_indices_flat
            ), 
            values_flat
        )
        
        return output_image
    
class Max_Pooling_Layer(Pooling_Layer):
    type = 'max_pool2d'

    def __init__(self,
            input_shape: tuple[int, int, int],
            kernel_size: tuple[int, int],
            padding: tuple[int, int],
            stride: tuple[int, int],
        ):
        
        super().__init__(
            input_shape=input_shape,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

    def max_pool2d(self, input: np.ndarray) -> np.ndarray:
        H_in, W_in, C_in = input.shape
        
        col_data, (out_h, out_w) = self.im2col(input)
        pooled_flat = np.max(col_data, axis=1)
        output_pooled = pooled_flat.reshape(out_h, out_w, C_in)

        self._arg_max_in_col_data: np.ndarray = np.argmax(col_data, axis=1) 

        return output_pooled

    def forward(self, input):
        self.input = input
        self.validate_input_shape()
        self.padded_input = self.get_padded_image(input)
        self.output = self.max_pool2d(self.padded_input)
        return self.output
    
    def backpropagation(self, derivatives: np.ndarray, learning_rate = None) -> np.ndarray:
        l_p = derivatives
        o_h, o_w, o_c = self.output_shape
        k_h, k_w = self.kernel_size
        p_h, p_w = self.padding
        i_h, i_w = self.input_shape
        
        derivatives_flat = l_p.reshape(o_h*o_w, o_c)

        gradient_col_data = np.zeros(
            (o_h*o_w , k_h*k_w , o_c),
            dtype=l_p.dtype
        )

        window_indices = np.arange(o_h*o_w)[:, np.newaxis] # Shape (out_h*out_w, 1)
        channel_indices = np.arange(o_c)[None, :]     # Shape (1, C_in)

        gradient_col_data[
            window_indices,             # For each window (row)# Shape (out_h*out_w, 1)
            self._arg_max_in_col_data,  # At the stored max index within the kernel elements# Shape (out_h*out_w, C_in)
            channel_indices             # For each channel (column)# Shape (1, C_in)
        ] = derivatives_flat
        
        dx_padded_image = self.col2im(gradient_col_data)

        l_x = dx_padded_image[
                                p_h : p_h + i_h,
                                p_w : p_w + i_w,
                                : 
                            ]
        
        return l_x

class Average_Pooling_Layer(Pooling_Layer):
    type = 'average_pool2d'

    def __init__(self,
            input_shape: tuple[int, int, int],
            kernel_size: tuple[int, int],
            padding: tuple[int, int],
            stride: tuple[int, int],
        ):
        
        super().__init__(
            input_shape=input_shape,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

    def average_pool2d(self, input: np.ndarray) -> np.ndarray:
        H_in, W_in, C_in = input.shape
        col_data, (out_h, out_w) = self.im2col(input)
        pooled_flat = np.mean(col_data, axis=1)
        output_pooled = pooled_flat.reshape(out_h, out_w, C_in)
        return output_pooled

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data # Store original input
        self.validate_input_shape() # Validate original input shape
        self.padded_input = self.get_padded_image(input_data) # Get padded version
        self.output = self.average_pool2d(self.padded_input) # Pass padded input to pooling operation
        return self.output
    
    def backpropagation(self, derivatives: np.ndarray, learning_rate = None) -> np.ndarray:
        l_p = derivatives # Incoming gradients, shape (out_h, out_w, C_in)
        o_h, o_w, o_c = self.output_shape
        k_h, k_w = self.kernel_size
        p_h, p_w = self.padding
        i_h, i_w, i_c = self.input_shape # Original input shape

        derivatives_flat = l_p.reshape(o_h * o_w, o_c) # Shape (NumWindows, C_in)

        kernel_area = k_h * k_w

        gradient_per_element_in_col = derivatives_flat / kernel_area # Shape (NumWindows, C_in)

        gradient_col_data = np.tile(
            gradient_per_element_in_col[:, np.newaxis, :], # Shape (NumWindows, 1, C_in)
            (1, kernel_area, 1)                           # Tile across the kernel_area dimension
        )
        
        dx_padded_image = self.col2im(gradient_col_data)

        dx_input = dx_padded_image[
                        p_h : p_h + i_h,
                        p_w : p_w + i_w,
                        : # all channels
                    ]
        
        return dx_input
    
class Flatten_Layer(Layer):
    type = 'flatten'
    
    def __init__(self, input_shape: tuple[int, int, int]):
        super().__init__(input_shape=input_shape)
        self.output_shape = (int(np.prod(input_shape)), )
    
    def flatten(self, input: np.ndarray):
        return input.flatten()
    
    def forward(self, input):
        self.input = input
        self.validate_input_shape()
        self.output = self.flatten(input)
        return self.output
    
    def backpropagation(self, derivatives, learning_rate=None):
        l_f = derivatives
        l_x = l_f.reshape(self.input_shape)
        return l_x
    
class Dense_Layer(Layer):
    type = 'dense'
    
    def __init__(self, 
            num_output: int,
            num_input: int = None,
            activation: Literal['relu', 'sigmoid', 'tanh', 'leaky_relu', 'softmax', 'indentity'] = 'relu',
        ):
        
        super().__init__(input_shape=(num_input,))
        
        self.output_shape = (num_output,)
        self.activation = activation
        self.activation_func = Activation_Functions.get(activation) 
        self.activation_derivative = Activation_Functions.get_derivative(activation)
        self.weights = np.random.randn(num_input, num_output)
        self.biases = np.random.randn(num_output)
    
    def _apply_weights(self, input: np.ndarray):
        result = input @ self.weights + self.biases
        return result
    
    def forward(self, input)-> np.ndarray:
        self.input = input
        self.validate_input_shape()
        self.z = self._apply_weights(input)
        self.output = self.activation_func(self.z)
        return self.output
    
    def backpropagation(self, derivatives, learning_rate=None):
        l_a = derivatives
        a_z = self.activation_derivative(self.output)

        l_z = l_a * a_z
        
        z_w = self.input
        l_w = np.outer(z_w, l_z)

        l_b = l_z.flatten()
        l_x = l_z @ self.weights.T

        self.weights -= learning_rate * l_w
        self.biases  -= learning_rate * l_b

        return l_x
    
    def backpropagation_with_softmax_and_cross_entropy(self, derivatives, learning_rate=None):
        l_z = derivatives
        
        z_w = self.input
        l_w = np.outer(z_w, l_z)

        l_b = l_z.flatten()
        l_x = l_z @ self.weights.T

        self.weights -= learning_rate * l_w
        self.biases  -= learning_rate * l_b

        return l_x

class Sequential:
    def __init__(self):
        self.layers : list[Layer] = []
        self.learning_rate = None

    def __repr__(self):
        return f"Model(name={self.name}, description={self.description}, model_type={self.model_type})"
    
    def _input_shape(self, input_shape: tuple[int, int, int] | None) -> tuple[int, int, int]:
        if self.layers:
            if input_shape:
                raise Exception('This is not first layer no need to provide input shape')
            else:
                input_shape = self.layers[-1].output_shape
        else:
            if input_shape:
                if len(input_shape) != 3:
                    raise Exception ("Input shape has to be of size 3. e.g. (28, 28, 1)")
            else:
                raise Exception('This is first layer you need to provide input shape')
            pass
            
        return input_shape
    
    @classmethod
    def _get_pooling_layer_maker(cls,
            pool_type: Literal['max_pool2d', 'average_pool2d']
        ) -> Callable :
        
        _registry = {
            'max_pool2d' : Max_Pooling_Layer,
            'average_pool2d' : Average_Pooling_Layer,
        }

        if pool_type not in _registry:
            raise Exception(f"{pool_type} is not in {_registry}")
            
        return _registry[pool_type]
        
    def convolutional_layer(self, 
            num_filters: int, 
            kernel_size: tuple[int, int],
            padding: tuple[int, int] = (0,0),
            stride: tuple[int, int] = (1,1),
            activation: Literal['relu', 'sigmoid', 'tanh', 'leaky_relu', 'softmax', 'identity'] = 'relu',
            input_shape: tuple[int, int, int] | None = None,
        ):
        input_shape = self._input_shape(input_shape)
        
        if len(input_shape) < 2:
            raise Exception(f"Convolutional layer requires input at least 2-dimensional but got {len(input_shape)}-dimension")
        
        layer = Filter_Layer(
            input_shape=input_shape,
            activation=activation,
            padding=padding,
            stride=stride,
            kernel_size=kernel_size,
            num_filters=num_filters,
        )
        
        self.layers.append(layer)
        
        print(f"{len(self.layers)}. Convolutional layer input_size: {input_shape} output_size: {layer.output_shape}")

    def pooling_layer(self, 
            kernel_size: tuple[int, int],
            padding: tuple[int, int],
            stride: tuple[int, int],
            pool_type: Literal['max_pool2d', 'average_pool2d'],
            input_shape: tuple[int, int, int] | None = None, 
        ):
        input_shape = self._input_shape(input_shape)
        
        if len(input_shape) < 2:
            raise Exception(f"Pooling layer requires input at least 2-dimensional but got {len(input_shape)}-dimension")
        
        pool_constructor = Sequential._get_pooling_layer_maker(pool_type)
        
        layer = pool_constructor(
            input_shape=input_shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        
        self.layers.append(layer)
        
        print(f"{len(self.layers)}. Pooling layer input_size: {input_shape} output_size: {layer.output_shape}")
    
    def flatten_layer(self):
        input_shape = self._input_shape(None)

        if len(input_shape) < 2:
            raise Exception(f"Flattening layer requires input at least 2-dimensional but got {len(input_shape)}-dimension")
        
        layer = Flatten_Layer(input_shape=input_shape)
            
        self.layers.append(layer)

        print(f"{len(self.layers)}. Flatten layer input_size: {input_shape} output_size: {layer.output_shape}")

    def dense_layer(self, 
            num_neurons: int = None, 
            activation: Literal['relu', 'sigmoid', 'tanh', 'leaky_relu', 'softmax', 'indentity'] = 'relu',
            input_shape: tuple[int, int, int] = None, 
        ):
        input_shape = self._input_shape(input_shape)
        
        layer = Dense_Layer(
            num_input=input_shape[0],
            num_output=num_neurons,
            activation=activation
        )
            
        self.layers.append(layer)

        print(f"{len(self.layers)}. Dense layer input_size: {input_shape} output_size: {layer.output_shape}")

    def forward_pass(self, input: np.ndarray):
        # print('FORWARD')
        first_layer = self.layers[0]
        if input.shape != first_layer.input_shape:
            raise Exception (f"Provided inputs shape is {input.shape} but expected {first_layer.input_shape}")

        for layer in self.layers:
            if (input.shape != layer.input_shape):
                raise Exception(f"{layer.type} layer says, Input shape is {input.shape} but expected {layer.input_shape}")

            input = layer.forward(input)
            # print(layer.type, layer.input.shape, layer.output.shape)
        
    def loss_function(self,
            function_name: Literal['mse','cross_entropy','binary_cross_entropy'],
        ):
        self.loss_type = function_name
        self.loss_func = Loss_Functions.get_loss_func(function_name)
        self.loss_derivative = Loss_Functions.get_derivative_func(function_name)
 
    def backpropogation(self,
            y_true: np.ndarray,
            learning_rate: float,
        ):
        # print('BACKWARD')
        last_layer = self.layers[-1]

        if not self.loss_func:
            raise Exception ('loss function is not set')
            
        loss = self.loss_func(y_true, last_layer.output)
        l_s = self.loss_derivative(y_true, last_layer.output)
        
        derivatives = l_s
        last_layer_flag = True
        for layer in reversed(self.layers):
            # print(layer.type,' output derivatives ',derivatives.shape)
            if last_layer_flag and layer.type == 'dense' and layer.activation == 'softmax' and self.loss_type == 'cross_entropy':
                last_layer_flag = False
                l_z = l_s
                derivatives = layer.backpropagation_with_softmax_and_cross_entropy(derivatives=l_z, learning_rate=learning_rate)
                continue
                
            derivatives = layer.backpropagation(derivatives, learning_rate=learning_rate)
        
        return loss

    def validate(self,
            y_true: np.ndarray,
        ):
        last_layer = self.layers[-1]
        loss = self.loss_func(y_true, last_layer.output)
        return loss
    
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        

        images = []
        for i in range(size):
            images.append([0] * rows * cols)
            
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(rows, cols)
            images[i][:] = img            
        
        return np.array(images), np.array(labels)
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        


# dense_layer = Dense_Layer(10, 8, 'relu')
# input = np.arange(8)
# y_true = np.arange(10)
# print(input)
# output = dense_layer.forward(input)
# print(output)

# loss = Loss_Functions.cross_entropy(y_true, output)
# print('Loss: ', loss)
# l_z = Loss_Functions.cross_entropy_derivative(y_true, output)

# grad = dense_layer.backpropagation_with_softmax_and_cross_entropy(derivatives=l_z, learning_rate=2)
# print(grad)






try:
    input_path = 'mnist'
    training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')
    # training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    # training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    # test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    # test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')


    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    input_shape = (*x_test[0].shape, 1)

    model = Sequential()

    model.convolutional_layer(
        input_shape=input_shape, 
        num_filters=6, 
        kernel_size=(5,5), 
        padding=(0,0), 
        stride=(1,1), 
        activation='tanh',
    )
    model.pooling_layer(
        kernel_size=(2,2), 
        padding=(0,0), 
        stride=(2,2),
        pool_type='average_pool2d',
    )
    model.convolutional_layer(
        num_filters=16, 
        kernel_size=(5,5), 
        padding=(0,0), 
        stride=(1,1), 
        activation='tanh',
    )
    model.pooling_layer(
        kernel_size=(2,2), 
        padding=(0,0), 
        stride=(2,2),
        pool_type='average_pool2d',
    )
    model.convolutional_layer(
        num_filters=120, 
        kernel_size=(4,4), 
        padding=(0,0), 
        stride=(1,1), 
        activation='tanh',
    )
    model.flatten_layer()
    model.dense_layer(
        num_neurons=84,
        activation='tanh'
    )
    model.dense_layer(
        num_neurons=10,
        activation='softmax',
    )
    
    model.loss_function(function_name='cross_entropy')
    
    epochs = 5
    mean = 0.1307
    std = 0.3081
    
    loss_cumulative = 0
    print('Learning')
    for epoch in range(epochs):
        i = 0
        for img, label in zip(x_train, y_train):
            img = img.astype(np.float32)
            img /= 225
            img = (img-mean)/std
            img = img[:,:,None]
            model.forward_pass(img)
            y_true = np.zeros((10,))
            y_true[label] = 1
            loss_cumulative += model.backpropogation(y_true=y_true, learning_rate=0.003)
            if i % 1000 == 0:
                print('epoch:', epoch, ' number: ', i, 'Loss: ', loss_cumulative/(i + epoch*len(y_train)))
            i +=1

    print(loss_cumulative/len(y_train))
    
    
    print('Validating')
    i = 0
    loss_cumulative = 0
    for img, label in zip(x_test, y_test):
        img = img.astype(np.float32)
        img /= 225
        img = (img-mean)/std
        
        img = img[:,:,None]
        model.forward_pass(img)
        
        y_true = np.zeros((10,))
        y_true[label] = 1
        loss_cumulative += model.validate(y_true=y_true)

        if i % 1000 == 0:
            print(' number: ', i, 'Lose: ', loss_cumulative/i)
        i += 1
        
        
    print(loss_cumulative/len(y_test))
except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()

    formatted_trace = traceback.extract_tb(exc_traceback)
    
    line_number = 'N/A' 
    if formatted_trace:
        last_call = formatted_trace[-1] 
        line_number = last_call.lineno

    print()
    print('Exception ::: ', e) 
    print(f'Error occurred on line: {line_number}') 
    print()
    traceback.print_exc() 




    