https://docs.pytorch.org/docs/stable/generated/torch.fft.hfftn.html

1.torch.fft.fft(input, n=None, dim=-1, norm=None, *, out=None) →

input (Tensor) – the input tensor

n (int, optional) – Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.

dim (int, optional) – The dimension along which to take the one dimensional FFT.

norm (str, optional) –

Normalization mode. For the forward transform (fft()), these correspond to:

"forward" - normalize by 1/n

"backward" - no normalization

"ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)

Calling the backward transform (ifft()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make ifft() the exact inverse.

Default is "backward" (no normalization)


2.torch.fft.ifft
torch.fft.ifft(input, n=None, dim=-1, norm=None, *, out=None) → Tensor



input (Tensor) – the input tensor

n (int, optional) – Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the IFFT.

dim (int, optional) – The dimension along which to take the one dimensional IFFT.

norm (str, optional) –

Normalization mode. For the backward transform (ifft()), these correspond to:

"forward" - no normalization

"backward" - normalize by 1/n

"ortho" - normalize by 1/sqrt(n) (making the IFFT orthonormal)

Calling the forward transform (fft()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make ifft() the exact inverse.

Default is "backward" (normalize by 1/n).

3.torch.fft.fft2
torch.fft.fft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) → Tensor


input (Tensor) – the input tensor

s (Tuple[int], optional) – Signal size in the transformed dimensions. If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the FFT. If a length -1 is specified, no padding is done in that dimension. Default: s = [input.size(d) for d in dim]

dim (Tuple[int], optional) – Dimensions to be transformed. Default: last two dimensions.

norm (str, optional) –

Normalization mode. For the forward transform (fft2()), these correspond to:

"forward" - normalize by 1/n

"backward" - no normalization

"ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)

Where n = prod(s) is the logical FFT size. Calling the backward transform (ifft2()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make ifft2() the exact inverse.

Default is "backward" (no normalization).

4。torch.fft.ifft2
torch.fft.ifft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) → Tensor

input (Tensor) – the input tensor

s (Tuple[int], optional) – Signal size in the transformed dimensions. If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the IFFT. If a length -1 is specified, no padding is done in that dimension. Default: s = [input.size(d) for d in dim]

dim (Tuple[int], optional) – Dimensions to be transformed. Default: last two dimensions.

norm (str, optional) –

Normalization mode. For the backward transform (ifft2()), these correspond to:

"forward" - no normalization

"backward" - normalize by 1/n

"ortho" - normalize by 1/sqrt(n) (making the IFFT orthonormal)

Where n = prod(s) is the logical IFFT size. Calling the forward transform (fft2()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make ifft2() the exact inverse.

Default is "backward" (normalize by 1/n).

5。torch.fft.fftn
torch.fft.fftn(input, s=None, dim=None, norm=None, *, out=None) → Tensor


input (Tensor) – the input tensor

s (Tuple[int], optional) – Signal size in the transformed dimensions. If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the FFT. If a length -1 is specified, no padding is done in that dimension. Default: s = [input.size(d) for d in dim]

dim (Tuple[int], optional) – Dimensions to be transformed. Default: all dimensions, or the last len(s) dimensions if s is given.

norm (str, optional) –

Normalization mode. For the forward transform (fftn()), these correspond to:

"forward" - normalize by 1/n

"backward" - no normalization

"ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)

Where n = prod(s) is the logical FFT size. Calling the backward transform (ifftn()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make ifftn() the exact inverse.

Default is "backward" (no normalization).

6。torch.fft.ifftn
torch.fft.ifftn(input, s=None, dim=None, norm=None, *, out=None) → Tensor
input (Tensor) – the input tensor

s (Tuple[int], optional) – Signal size in the transformed dimensions. If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the IFFT. If a length -1 is specified, no padding is done in that dimension. Default: s = [input.size(d) for d in dim]

dim (Tuple[int], optional) – Dimensions to be transformed. Default: all dimensions, or the last len(s) dimensions if s is given.

norm (str, optional) –

Normalization mode. For the backward transform (ifftn()), these correspond to:

"forward" - no normalization

"backward" - normalize by 1/n

"ortho" - normalize by 1/sqrt(n) (making the IFFT orthonormal)

Where n = prod(s) is the logical IFFT size. Calling the forward transform (fftn()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make ifftn() the exact inverse.

Default is "backward" (normalize by 1/n).

7.
.fft.rfft
torch.fft.rfft(input, n=None, dim=-1, norm=None, *, out=None) → Tensor
input (Tensor) – the real input tensor

n (int, optional) – Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the real FFT.

dim (int, optional) – The dimension along which to take the one dimensional real FFT.

norm (str, optional) –

Normalization mode. For the forward transform (rfft()), these correspond to:

"forward" - normalize by 1/n

"backward" - no normalization

"ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)

Calling the backward transform (irfft()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make irfft() the exact inverse.

Default is "backward" (no normalization).

Keyword Arguments

8.；torch.fft.irfft
torch.fft.irfft(input, n=None, dim=-1, norm=None, *, out=None) → Tensor


input (Tensor) – the input tensor representing a half-Hermitian signal

n (int, optional) – Output signal length. This determines the length of the output signal. If given, the input will either be zero-padded or trimmed to this length before computing the real IFFT. Defaults to even output: n=2*(input.size(dim) - 1).

dim (int, optional) – The dimension along which to take the one dimensional real IFFT.

norm (str, optional) –

Normalization mode. For the backward transform (irfft()), these correspond to:

"forward" - no normalization

"backward" - normalize by 1/n

"ortho" - normalize by 1/sqrt(n) (making the real IFFT orthonormal)

Calling the forward transform (rfft()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make irfft() the exact inverse.

Default is "backward" (normalize by 1/n)
9.torch.fft.rfft2
torch.fft.rfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) → Tensor

input (Tensor) – the input tensor

s (Tuple[int], optional) – Signal size in the transformed dimensions. If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the real FFT. If a length -1 is specified, no padding is done in that dimension. Default: s = [input.size(d) for d in dim]

dim (Tuple[int], optional) – Dimensions to be transformed. Default: last two dimensions.

norm (str, optional) –

Normalization mode. For the forward transform (rfft2()), these correspond to:

"forward" - normalize by 1/n

"backward" - no normalization

"ortho" - normalize by 1/sqrt(n) (making the real FFT orthonormal)

Where n = prod(s) is the logical FFT size. Calling the backward transform (irfft2()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make irfft2() the exact inverse.

Default is "backward" (no normalization).

Keyword Arguments

10.torch.fft.irfft2
torch.fft.irfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) → Tensor

input (Tensor) – the input tensor

s (Tuple[int], optional) – Signal size in the transformed dimensions. If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the real FFT. If a length -1 is specified, no padding is done in that dimension. Defaults to even output in the last dimension: s[-1] = 2*(input.size(dim[-1]) - 1).

dim (Tuple[int], optional) – Dimensions to be transformed. The last dimension must be the half-Hermitian compressed dimension. Default: last two dimensions.

norm (str, optional) –

Normalization mode. For the backward transform (irfft2()), these correspond to:

"forward" - no normalization

"backward" - normalize by 1/n

"ortho" - normalize by 1/sqrt(n) (making the real IFFT orthonormal)

Where n = prod(s) is the logical IFFT size. Calling the forward transform (rfft2()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make irfft2() the exact inverse.

Default is "backward" (normalize by 1/n).

Keyword Arguments

11.torch.fft.rfftn
torch.fft.rfftn(input, s=None, dim=None, norm=None, *, out=None)
input (Tensor) – the input tensor

s (Tuple[int], optional) – Signal size in the transformed dimensions. If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the real FFT. If a length -1 is specified, no padding is done in that dimension. Default: s = [input.size(d) for d in dim]

dim (Tuple[int], optional) – Dimensions to be transformed. Default: all dimensions, or the last len(s) dimensions if s is given.

norm (str, optional) –

Normalization mode. For the forward transform (rfftn()), these correspond to:

"forward" - normalize by 1/n

"backward" - no normalization

"ortho" - normalize by 1/sqrt(n) (making the real FFT orthonormal)

Where n = prod(s) is the logical FFT size. Calling the backward transform (irfftn()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make irfftn() the exact inverse.

Default is "backward" (no normalization).

Keyword Arguments
out (Tensor, optional) – the output tensor.


12.torch.fft.irfftn
torch.fft.irfftn(input, s=None, dim=None, norm=None, *, out=None) → Tensor


input (Tensor) – the input tensor

s (Tuple[int], optional) – Signal size in the transformed dimensions. If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the real FFT. If a length -1 is specified, no padding is done in that dimension. Defaults to even output in the last dimension: s[-1] = 2*(input.size(dim[-1]) - 1).

dim (Tuple[int], optional) – Dimensions to be transformed. The last dimension must be the half-Hermitian compressed dimension. Default: all dimensions, or the last len(s) dimensions if s is given.

norm (str, optional) –

Normalization mode. For the backward transform (irfftn()), these correspond to:

"forward" - no normalization

"backward" - normalize by 1/n

"ortho" - normalize by 1/sqrt(n) (making the real IFFT orthonormal)

Where n = prod(s) is the logical IFFT size. Calling the forward transform (rfftn()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make irfftn() the exact inverse.

Default is "backward" (normalize by 1/n).

13.torch.fft.hfft
torch.fft.hfft(input, n=None, dim=-1, norm=None, *, out=None) → Tensor

input (Tensor) – the input tensor representing a half-Hermitian signal

n (int, optional) – Output signal length. This determines the length of the real output. If given, the input will either be zero-padded or trimmed to this length before computing the Hermitian FFT. Defaults to even output: n=2*(input.size(dim) - 1).

dim (int, optional) – The dimension along which to take the one dimensional Hermitian FFT.

norm (str, optional) –

Normalization mode. For the forward transform (hfft()), these correspond to:

"forward" - normalize by 1/n

"backward" - no normalization

"ortho" - normalize by 1/sqrt(n) (making the Hermitian FFT orthonormal)

Calling the backward transform (ihfft()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make ihfft() the exact inverse.

Default is "backward" (no normalization).

14.torch.fft.ihfft
torch.fft.ihfft(input, n=None, dim=-1, norm=None, *, out=None) → Tensor

input (Tensor) – the real input tensor

n (int, optional) – Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the Hermitian IFFT.

dim (int, optional) – The dimension along which to take the one dimensional Hermitian IFFT.

norm (str, optional) –

Normalization mode. For the backward transform (ihfft()), these correspond to:

"forward" - no normalization

"backward" - normalize by 1/n

"ortho" - normalize by 1/sqrt(n) (making the IFFT orthonormal)

Calling the forward transform (hfft()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make ihfft() the exact inverse.

Default is "backward" (normalize by 1/n).

15.torch.fft.hfft2
torch.fft.hfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) → Tensor

input (Tensor) – the input tensor

s (Tuple[int], optional) – Signal size in the transformed dimensions. If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the Hermitian FFT. If a length -1 is specified, no padding is done in that dimension. Defaults to even output in the last dimension: s[-1] = 2*(input.size(dim[-1]) - 1).

dim (Tuple[int], optional) – Dimensions to be transformed. The last dimension must be the half-Hermitian compressed dimension. Default: last two dimensions.

norm (str, optional) –

Normalization mode. For the forward transform (hfft2()), these correspond to:

"forward" - normalize by 1/n

"backward" - no normalization

"ortho" - normalize by 1/sqrt(n) (making the Hermitian FFT orthonormal)

Where n = prod(s) is the logical FFT size. Calling the backward transform (ihfft2()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make ihfft2() the exact inverse.

Default is "backward" (no normalization).

16.torch.fft.ihfft2
torch.fft.ihfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) → Tensor

input (Tensor) – the input tensor

s (Tuple[int], optional) – Signal size in the transformed dimensions. If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the Hermitian IFFT. If a length -1 is specified, no padding is done in that dimension. Default: s = [input.size(d) for d in dim]

dim (Tuple[int], optional) – Dimensions to be transformed. Default: last two dimensions.

norm (str, optional) –

Normalization mode. For the backward transform (ihfft2()), these correspond to:

"forward" - no normalization

"backward" - normalize by 1/n

"ortho" - normalize by 1/sqrt(n) (making the Hermitian IFFT orthonormal)

Where n = prod(s) is the logical IFFT size. Calling the forward transform (hfft2()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make ihfft2() the exact inverse.

Default is "backward" (normalize by 1/n).

17.torch.fft.hfftn
torch.fft.hfftn(input, s=None, dim=None, norm=None, *, out=None) → Tensor

input (Tensor) – the input tensor

s (Tuple[int], optional) – Signal size in the transformed dimensions. If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the real FFT. If a length -1 is specified, no padding is done in that dimension. Defaults to even output in the last dimension: s[-1] = 2*(input.size(dim[-1]) - 1).

dim (Tuple[int], optional) – Dimensions to be transformed. The last dimension must be the half-Hermitian compressed dimension. Default: all dimensions, or the last len(s) dimensions if s is given.

norm (str, optional) –

Normalization mode. For the forward transform (hfftn()), these correspond to:

"forward" - normalize by 1/n

"backward" - no normalization

"ortho" - normalize by 1/sqrt(n) (making the Hermitian FFT orthonormal)

Where n = prod(s) is the logical FFT size. Calling the backward transform (ihfftn()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make ihfftn() the exact inverse.

Default is "backward" (no normalization).



18.torch.fft.ihfftn
torch.fft.ihfftn(input, s=None, dim=None, norm=None, *, out=None) → Tensor

input (Tensor) – the input tensor

s (Tuple[int], optional) – Signal size in the transformed dimensions. If given, each dimension dim[i] will either be zero-padded or trimmed to the length s[i] before computing the Hermitian IFFT. If a length -1 is specified, no padding is done in that dimension. Default: s = [input.size(d) for d in dim]

dim (Tuple[int], optional) – Dimensions to be transformed. Default: all dimensions, or the last len(s) dimensions if s is given.

norm (str, optional) –

Normalization mode. For the backward transform (ihfftn()), these correspond to:

"forward" - no normalization

"backward" - normalize by 1/n

"ortho" - normalize by 1/sqrt(n) (making the Hermitian IFFT orthonormal)

Where n = prod(s) is the logical IFFT size. Calling the forward transform (hfftn()) with the same normalization mode will apply an overall normalization of 1/n between the two transforms. This is required to make ihfftn() the exact inverse.

Default is "backward" (normalize by 1/n).

19.torch.fft.fftfreq
torch.fft.fftfreq(n, d=1.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

n (int) – the FFT length

d (float, optional) – The sampling length scale. The spacing between individual samples of the FFT input. The default assumes unit spacing, dividing that result by the actual spacing gives the result in physical frequency units.

Keyword Arguments
out (Tensor, optional) – the output tensor.

dtype (torch.dtype, optional) – the desired data type of returned tensor. Default: if None, uses a global default (see torch.set_default_dtype()).

layout (torch.layout, optional) – the desired layout of returned Tensor. Default: torch.strided.

device (torch.device, optional) – the desired device of returned tensor. Default: if None, uses the current device for the default tensor type (see torch.set_default_device()). device will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

requires_grad (bool, optional) – If autograd should record operations on the returned tensor. Default: False.

Example

20.torch.fft.rfftfreq
torch.fft.rfftfreq(n, d=1.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
n (int) – the real FFT length

d (float, optional) – The sampling length scale. The spacing between individual samples of the FFT input. The default assumes unit spacing, dividing that result by the actual spacing gives the result in physical frequency units.

Keyword Arguments
out (Tensor, optional) – the output tensor.

dtype (torch.dtype, optional) – the desired data type of returned tensor. Default: if None, uses a global default (see torch.set_default_dtype()).

layout (torch.layout, optional) – the desired layout of returned Tensor. Default: torch.strided.

device (torch.device, optional) – the desired device of returned tensor. Default: if None, uses the current device for the default tensor type (see torch.set_default_device()). device will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

requires_grad (bool, optional) – If autograd should record operations on the returned tensor. Default: False.

21.torch.fft.fftshift
torch.fft.fftshift(input, dim=None) → Tensor
Reorders n-dimensional FFT data, as provided by fftn(), to have negative frequency terms first.

This performs a periodic shift of n-dimensional data such that the origin (0, ..., 0) is moved to the center of the tensor. Specifically, to input.shape[dim] // 2 in each selected dimension.

input (Tensor) – the tensor in FFT order

dim (int, Tuple[int], optional) – The dimensions to rearrange. Only dimensions specified here will be rearranged, any other dimensions will be left in their original order. Default: All dimensions of input.

Example

22.torch.fft.ifftshift
torch.fft.ifftshift(input, dim=None) → Tensor

input (Tensor) – the tensor in FFT order

dim (int, Tuple[int], optional) – The dimensions to rearrange. Only dimensions specified here will be rearranged, any other dimensions will be left in their original order. Default: All dimensions of input.

Example