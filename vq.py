import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from typing import *
import argparse
import os


###
### Implements Lloyd's Algorithm for Vector Quantization
### Trained via fit(...), inferences / quantizes via forward(...)
###
class VectorQuantizer(nn.Module):
	def __init__(self, codebook_size: int, channels: int, block_size: int, max_iters: int):
		super(VectorQuantizer, self).__init__()
		self.codebook_size = codebook_size
		self.block_size = block_size
		self.max_iters = max_iters
		self.codebook = torch.zeros(size=[codebook_size, block_size * block_size, channels])
			
	
	### Deconstruct image into blocks
	### Input (H, W, C) -> Output (L, B*B, C)
	def blockify_image(self, image: torch.Tensor) -> torch.Tensor:
		B = self.block_size
		H, W, C = image.shape
		blocks = image.unfold(0, B, B).unfold(1, B, B)
		blocks = blocks.reshape(-1, self.block_size * self.block_size, C)
		return blocks


	### Reconstruct from quantized codebook blocks
	### Input (L, B*B, C) -> Output (H, W, C)
	### TODO: Fix to work with color?
	def unblockify_image(self, blocks: torch.Tensor, image_shape: Tuple[int, int]) -> torch.Tensor:
		H, W, C = image_shape
		B = self.block_size
		blocks = blocks.view(H // B, W // B, B, B, C)
		blocks = blocks.permute(0, 2, 1, 3, 4).contiguous()
		return blocks.view(H, W, C)


	def _codebook_block_distance(self, blocks: torch.Tensor) -> torch.Tensor:
		L, BB, C = blocks.shape
		blocks = blocks.unsqueeze(1).expand(L, self.codebook_size, BB, C)
		codebook_expanded = self.codebook.unsqueeze(0).expand(L, self.codebook_size, BB, C)
		distances = torch.linalg.vector_norm(blocks - codebook_expanded, dim=2)
		return distances	
	

	### Lloyd's Algorithm to create a codebook
	def fit(self, tensor_dataset: List[torch.Tensor]) -> Any:
		### Blockify the tensor dataset
		all_blocks = torch.cat([self.blockify_image(img) for img in tensor_dataset], dim=0)

		### Iterate until convergence or maximum iterations reached
		for iteration_index in range(self.max_iters):
			distances = self._codebook_block_distance(all_blocks)
			closest = torch.argmin(distances, dim=1)
			new_codebook = torch.zeros_like(self.codebook)
			counts = torch.zeros(self.codebook_size)

			print(f"Iteration {iteration_index+1}/{self.max_iters}")
			# print(f"Distances Shape: {distances.shape}")
			# print(f"Closest Shape: {closest.shape}")

			### Populate new codebook
			for i in range(all_blocks.shape[0]):
				closest_indices = closest[i]
				new_codebook[closest_indices] += all_blocks[i]
				counts[closest_indices] += 1

			### Avoid division by zero, rescale codebook by counts
			for i in range(self.codebook_size):
				if counts[i] > 0:
					new_codebook[i] /= counts[i]

			### Check for convergence - if we haven't changed much, then stop early!
			if torch.allclose(self.codebook, new_codebook, atol=1e-5):
				print(f"Early exiting! Converged early")
				break

			self.codebook = new_codebook

		return self


	### Internal function that actually does the quantization
	def _quantize(self, image: torch.Tensor) -> torch.Tensor:
		blocks = self.blockify_image(image)
		distances = self._codebook_block_distance(blocks)
		closest = torch.argmin(distances, dim=1)
		return self.codebook[closest]


	### Apply quantization to an image
	def forward(self, image: torch.Tensor) -> torch.Tensor:
		blocks = self.blockify_image(image)
		quantized_blocks = self._quantize(blocks)
		return self.unblockify_image(quantized_blocks, image.shape)


if __name__ == "__main__":
	### Get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--image-folder", type=str, default="train/")
	parser.add_argument("--image-eval-folder", type=str, default="eval/")
	parser.add_argument("--output-folder", type=str, default="output/")
	parser.add_argument("--codebook-size", type=int, default=128)
	parser.add_argument("--block-size", type=int, default=4)
	parser.add_argument("--max-iters", type=int, default=4)
	args = parser.parse_args()

	### Variable Init
	image_train_tensors = []
	image_eval_tensors = []

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.CenterCrop([256, 256]),
		transforms.Resize([224, 224]),
	])

	to_pil_transform = transforms.Compose([
		transforms.ToPILImage()
	])

	### Cache training images
	### If dataset is large enough, I wouldn't do this obviously.
	for filename in os.listdir(args.image_folder):
		if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
			print(f"Loading {filename} for training split...")
			image_path = os.path.join(args.image_folder, filename)
			image = Image.open(image_path).convert("L")
			image_tensor = transform(image).permute(1, 2, 0)
			image_train_tensors.append(image_tensor)

	print(f"Fitting Vector Quantizer with {len(image_train_tensors)} images...")

	### Initialize and train/fit vector quantizer
	vq = VectorQuantizer(
		codebook_size=args.codebook_size, 
		channels=1, 
		block_size=args.block_size, 
		max_iters=args.max_iters
	).fit(image_train_tensors)

	### Quantize and reconstruct images
	for filename in os.listdir(args.image_eval_folder):
		if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
			print(f"Loading {filename} from eval split for quantization")
			image_path = os.path.join(args.image_folder, filename)
			image = Image.open(image_path).convert("L")
			image_tensor = transform(image).permute(1, 2, 0)
			### Pass 'image_tensor' through vector quantizer 'vq'
			### Afterwards, convert to (C, H, W) format instead of (H, W, C) to save as PIL image
			quantized_image_tensor = vq(image_tensor).permute(2, 0, 1)
			quantized_image = to_pil_transform(quantized_image_tensor)
			quantized_image.save(os.path.join(args.output_folder, f"quant_c{args.codebook_size}_b{args.block_size}_{filename}"))
	

