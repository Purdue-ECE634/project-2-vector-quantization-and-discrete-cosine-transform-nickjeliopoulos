import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from typing import *
import argparse
import os


def PSNR(source, target) -> float:
	### Numerical stability (what if mse = 0?)
	epsilon = 1e-9
	### Compute Mean Squared Error (MSE), max pixel value (255.0 or 1.0 depending on how the image is loaded)
	mse = np.mean((source - target) ** 2)
	max_pixel = np.max(target)
	psnr = 20 * np.log10(max_pixel / (np.sqrt(mse) + epsilon))
	return psnr


def dct2(block):
	"""Compute the 2D Discrete Cosine Transform (DCT) of a block."""
	return scipy.fftpack.dct(scipy.fftpack.dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(block):
	"""Compute the 2D Inverse Discrete Cosine Transform (IDCT) of a block."""
	return scipy.fftpack.idct(scipy.fftpack.idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')


def apply_dct(image, block_size=8):
	""" Apply block-wise DCT to an image. """
	h, w = image.shape
	dct_image = np.zeros((h, w))
	
	for i in range(0, h, block_size):
		for j in range(0, w, block_size):
			dct_image[i:i+block_size, j:j+block_size] = dct2(image[i:i+block_size, j:j+block_size])
	
	return dct_image


def apply_topk_magnitude_idct(dct_image, block_size=8, k=8):
	"""	Reconstruct image using only largest mangitude coefficients to reconstruct the image
	"""
	h, w = dct_image.shape
	reconstructed = np.zeros((h, w))
	
	for i in range(0, h, block_size):
		for j in range(0, w, block_size):
			block = dct_image[i:i+block_size, j:j+block_size].copy()
			
			### Keep only the top-k largest magnitude coefficients
			### Notice the -np.abs() to sort in descending order
			flat_indices = np.argsort(-np.abs(block).flatten())
			### Mask out everything except these largest top-K coefficients
			mask = np.zeros_like(block)
			mask.flat[flat_indices[:k]] = 1
			block *= mask
			
			reconstructed[i:i+block_size, j:j+block_size] = idct2(block)
	
	return reconstructed


def get_causal_mask(block_size: int, r: int) -> Tuple[np.ndarray, int]:
	"""Create a causal mask for the IDCT reconstruction"""
	mask = np.zeros((block_size, block_size))

	for i in range(block_size):
		for j in range(block_size):
			do_mask = i + j < block_size*block_size and i + j < r
			mask[i, j] = 1 if do_mask else 0

	return mask


def apply_firstk_causal_index_idct(dct_image, block_size=8, k=8):
	"""	Reconstruct image using a causal mask over DCT coefficients.
		Note "k" in this context is slightly different from the previous function.
		The mask is causal - see https://www.mathworks.com/help/images/discrete-cosine-transform.html for a similar implementation
	"""
	h, w = dct_image.shape
	reconstructed = np.zeros((h, w))
	mask = get_causal_mask(block_size, k)
	print(f"Causal Mask ({np.sum(mask, dtype=np.int32)} non-zero):\n{mask}")
	
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
	parser.add_argument("--output-folder", type=str, default="output/")
	args = parser.parse_args()

	### Load and preprocess image
	image = Image.open(args.image_path).convert("L")

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.CenterCrop([256, 256]),
		transforms.Resize([224, 224]),
	])

	image_tensor = transform(image).squeeze().numpy()
	
	### Apply DCT and reconstruct with different coefficients
	dct_image = apply_dct(image_tensor, block_size=8)
	print(f"Original image shape: {image_tensor.shape}")
	print(f"DCT image shape: {dct_image.shape}")
	
	###
	### Top-K Magnitude
	###
	plt.figure(figsize=(10, 5))
	for idx, k in enumerate([2, 4, 8, 16, 32]):
		### Switch out IDCT variant here
		reconstructed = apply_topk_magnitude_idct(dct_image, block_size=8, k=k)
		plt.subplot(2, 3, idx+1)
		plt.imshow(reconstructed, cmap='gray')
		plt.title(f'K={k} | PSNR={PSNR(image_tensor, reconstructed):.1f}')
		plt.axis('off')
		path = os.path.join(args.output_folder, f'dct_reconstructed_topk_mag_k{k}.png')
		plt.imsave(path, reconstructed, cmap='gray')

	plt.subplot(2, 3, 6)
	plt.imshow(image_tensor, cmap='gray')
	plt.title('Original')
	plt.axis('off')
	plt.show()

	###
	### First-K Causal Index Mask
	###
	plt.figure(figsize=(10, 5))
	for idx, k in enumerate([1, 2, 3, 6, 8]):
		### Switch out IDCT variant here
		reconstructed = apply_firstk_causal_index_idct(dct_image, block_size=8, k=k)
		plt.subplot(2, 3, idx+1)
		plt.imshow(reconstructed, cmap='gray')
		plt.title(f'K={k} | PSNR={PSNR(image_tensor, reconstructed):.1f}')
		plt.axis('off')
		path = os.path.join(args.output_folder, f'dct_reconstructed_causal_k{k}.png')
		plt.imsave(path, reconstructed, cmap='gray')
	
	plt.subplot(2, 3, 6)
	plt.imshow(image_tensor, cmap='gray')
	plt.title('Original')
	plt.axis('off')
	plt.show()
	