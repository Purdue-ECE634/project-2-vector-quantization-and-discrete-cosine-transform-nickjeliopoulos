import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from typing import *
import argparse
import os

plt.style.use('fast')

def PSNR(source, target) -> float:
	### Numerical stability (what if mse = 0?)
	epsilon = 1e-9
	### Compute Mean Squared Error (MSE), max pixel value (255.0 or 1.0 depending on how the image is loaded)
	mse = np.mean((source - target) ** 2)
	max_pixel = np.max(target)
	psnr = 20 * np.log10(max_pixel / (np.sqrt(mse) + epsilon))
	return psnr


### Rather than using PyTorch, I just used the minimal scipy implementation for DCT
### A bit easier this way. I could have converted the output to Torch, but it's not necessary becaus the script is so short anyway.
def dct2(block):
	"""Compute the 2D Discrete Cosine Transform (DCT) of a block."""
	return scipy.fftpack.dct(scipy.fftpack.dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(block):
	"""Compute the 2D Inverse Discrete Cosine Transform (IDCT) of a block."""
	return scipy.fftpack.idct(scipy.fftpack.idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')


### Used for the causal mask
def zigzag_indices(n: int) -> List[Tuple[int, int]]:
    indices = [(x, y) for x in range(n) for y in range(n)]
    indices.sort(key=lambda x: (x[0] + x[1], x[1] if (x[0] + x[1]) % 2 == 0 else -x[1]))
    return indices


### Generate mask for IDCT reconstruction, where only the first-k coefficients are kept
def get_causal_mask(block_size: int, r: int) -> np.ndarray:
    """Generate a causal mask for the first-k DCT coefficients using zig-zag indexing."""
    mask = np.zeros((block_size, block_size), dtype=np.float32)
    indices = zigzag_indices(block_size)
    for idx in range(k):
        x, y = indices[idx]
        mask[x, y] = 1.0
    return mask


### Forward DCT
def apply_dct(image, block_size=8):
	""" Apply block-wise DCT to an image. """
	h, w = image.shape
	dct_image = np.zeros((h, w))
	
	for i in range(0, h, block_size):
		for j in range(0, w, block_size):
			dct_image[i:i+block_size, j:j+block_size] = dct2(image[i:i+block_size, j:j+block_size])
	
	return dct_image


### Inverse DCT (IDCT) with a causal mask
def apply_firstk_causal_index_idct(dct_image, block_size=8, k=8):
	"""	Reconstruct image using a causal mask over DCT coefficients.
		Note "k" in this context is slightly different from the previous function.
		The mask is causal - see https://www.mathworks.com/help/images/discrete-cosine-transform.html for a similar implementation
	"""
	h, w = dct_image.shape
	reconstructed = np.zeros((h, w))
	mask = get_causal_mask(block_size, k)
	# print(f"Causal Mask ({np.sum(mask, dtype=np.int32)} non-zero):\n{mask}")
	
	for i in range(0, h, block_size):
		for j in range(0, w, block_size):
			### Get our block, then apply the causal mask
			block = dct_image[i:i+block_size, j:j+block_size].copy()
			block *= mask
			
			reconstructed[i:i+block_size, j:j+block_size] = idct2(block)
	
	return reconstructed


if __name__ == "__main__":
	### Get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--image-path", type=str, default="akiyo0000.png")
	parser.add_argument("--block-size", type=int, default=8)
	parser.add_argument("--output-folder", type=str, default="output/")
	args = parser.parse_args()

	### Load and preprocess image
	image = Image.open(args.image_path).convert("L")

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.CenterCrop([256, 256]),
		transforms.Resize([224, 224]),
	])

	original_image = transform(image).squeeze().numpy()
	
	### Apply DCT and reconstruct with different coefficients
	dct_image = apply_dct(original_image, block_size=args.block_size)
	print(f"Original image shape: {original_image.shape}")
	print(f"DCT image shape: {dct_image.shape}")

	###
	### First-K Causal Index Mask
	###
	k_list = [1, 2, 4, 8, 16, 32, 48, 63, 64]
	plot_columns = 3
	plot_rows = 3

	plt.figure(figsize=(16, 9))

	for idx, k in enumerate(k_list):
		### Switch out IDCT variant here
		reconstructed = apply_firstk_causal_index_idct(dct_image, block_size=args.block_size, k=k)
		plt.subplot(plot_rows, plot_columns, idx+1)
		plt.imshow(reconstructed, cmap='gray')
		plt.title(f'K={k} | PSNR={PSNR(original_image, reconstructed):.1f}')
		plt.axis('off')
		path = os.path.join(args.output_folder, f'dct_reconstructed_causal_k{k}.png')
		plt.imsave(path, reconstructed, cmap='gray')

	plt.tight_layout()
	plt.show()
	