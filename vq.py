import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from typing import *
import argparse
import os


###
### Trained via fit(...), inferences / quantizes via forward(...)
###
class VectorQuantizer(nn.Module):
	def __init__(self, codebook_size: int, channels: int, block_size: int, max_iters: int):
		super(VectorQuantizer, self).__init__()
		self.codebook_size = codebook_size
		self.block_size = block_size
		self.max_iters = max_iters
		self.codebook = torch.empty(size=[codebook_size, block_size * block_size, channels])
			
	
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
	def unblockify_image(self, blocks: torch.Tensor, image_shape: Tuple[int, int]) -> torch.Tensor:
		H, W, C = image_shape
		B = self.block_size
		blocks = blocks.view(H // B, W // B, B, B, C)
		blocks = blocks.permute(0, 2, 1, 3, 4).contiguous()
		return blocks.view(H, W, C)


	### Identify the closest codebook vector for each block
	### Returns (L, codebook_size) tensor of distances
	### Interpretation: each block (l in L) has a distance to each codebook vector (n in codebook_size)
	def _codebook_block_distance(self, blocks: torch.Tensor) -> torch.Tensor:
		L, BB, C = blocks.shape
		N, _, _ = self.codebook.shape
		### Flatten blocks   (L, B*B, C) -> (L, B*B*C)
		### Flatten codebook (N, B*B, C) -> (N, B*B*C)
		blocks_flat = blocks.reshape(L, -1)
		codebook_flat = self.codebook.reshape(N, -1)
		### Use p-norm (p=2) as a distance metric
		distances = torch.cdist(blocks_flat, codebook_flat, p=2)
		### NOTE: You could apply 1.0 - softmax(x) on the distance metric to get a probability distribution
		### You could sample from this for fitting - might regularize?
		return distances
	

	### Fit codebook to dataset (could be a single image, or many!)
	@torch.inference_mode()
	def fit(self, tensor_train_dataset: List[torch.Tensor]) -> Any:
		### Blockify the tensor dataset
		block_train_dataset = torch.cat([self.blockify_image(img) for img in tensor_train_dataset], dim=0)
		N, BB, C = block_train_dataset.shape

		### Initialize codebook with N random blocks from the block_train_dataset
		indices = torch.randperm(N)[:self.codebook_size]
		self.codebook = block_train_dataset[indices]

		### Iterate until convergence or maximum iterations reached
		for iteration_index in range(self.max_iters):
			### Compute distances between each block and each codebook vector, then find the closest codebook vector to each block
			block_codebook_distances = self._codebook_block_distance(block_train_dataset)
			closest_codebook_indices = torch.argmin(block_codebook_distances, dim=1)

			### Initialize new codebook and counts
			### NOTE: There are many choices for initialization, this is just one
			### Add a small amount of noise to the codebook. Because we initialize the codebook with random blocks from the actual trainset, we want to encourage
			### more variance in the codebook. This changes some of the closest distance computations, and should result in more diverse codebook vectors.
			### Unsure. Leaving out.
			new_codebook = self.codebook.clone() # + torch.randn_like(self.codebook) * 1e-4

			### Old (SLOW!)
			### Populate new codebook by summing up all blocks that are closest to each codebook vector
			### Use a running average to update the codebook in a single loop
			# counts = torch.zeros(N, dtype=torch.float32)
			# for i in range(N):
			# 	closest_index = closest_codebook_indices[i]
			# 	counts[closest_index] += 1.0
			# 	new_codebook[closest_index] += (block_train_dataset[i] - new_codebook[closest_index]) / counts[closest_index]

			### New. Faster with the vectorized mean
			for idx in range(self.codebook_size):
				assigned_blocks = block_train_dataset[closest_codebook_indices == idx]
				### If we have assigned blocks, then update the codebook vector
				if len(assigned_blocks) > 0:
					new_codebook[idx] = assigned_blocks.mean(dim=0)
				### Empty clusters? - just keep the old codebook vector
				### NOTE: We can just pass here, if we initialize the new_codebook with the old codebook
				else:
					pass
					#new_codebook[idx] = self.codebook[idx]

			### Check for convergence - if we haven't changed much, then stop early!
			if torch.allclose(self.codebook, new_codebook, rtol=1e-6, atol=1e-8):
				print(f"Early exiting - converged!")
				break
			
			### Update codebook
			self.codebook = new_codebook

		### Info at the end for fun
		print(f"Finished fitting codebook in {iteration_index+1} iterations")

		return self


	### Internal function that actually does the quantization
	def _quantize(self, image: torch.Tensor) -> torch.Tensor:
		blocks = self.blockify_image(image)
		distances = self._codebook_block_distance(blocks)
		closest = torch.argmin(distances, dim=1)
		return self.codebook[closest]


	### Apply quantization to an image
	@torch.inference_mode()
	def forward(self, image: torch.Tensor) -> torch.Tensor:
		blocks = self.blockify_image(image)
		quantized_blocks = self._quantize(blocks)
		return self.unblockify_image(quantized_blocks, image.shape)


if __name__ == "__main__":
	### Set seeds. We do use a randn(...) in here
	torch.manual_seed(37)
	np.random.seed(37)

	### Get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--image-folder", type=str, default="train/")
	parser.add_argument("--image-eval-folder", type=str, default="eval/")
	parser.add_argument("--output-folder", type=str, default="output/")
	parser.add_argument("--codebook-size", type=int, default=128)
	parser.add_argument("--block-size", type=int, default=4)
	parser.add_argument("--max-iters", type=int, default=128)
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
			image_path = os.path.join(args.image_eval_folder, filename)
			image = Image.open(image_path).convert("L")
			image_tensor = transform(image).permute(1, 2, 0)
			### Pass 'image_tensor' through vector quantizer 'vq'
			### Afterwards, convert to (C, H, W) format instead of (H, W, C) to save as PIL image
			quantized_image_tensor = vq(image_tensor).permute(2, 0, 1)
			quantized_image = to_pil_transform(quantized_image_tensor)
			quantized_image.save(os.path.join(args.output_folder, f"quant_c{args.codebook_size}_b{args.block_size}_{filename}"))
	
	