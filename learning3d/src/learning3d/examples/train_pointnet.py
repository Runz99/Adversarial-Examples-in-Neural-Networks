import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
import time
import torchattacks
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
	sys.path.append(os.path.join(BASE_DIR, os.pardir))
	os.chdir(os.path.join(BASE_DIR, os.pardir))
	
from learning3d.models import PointNet
from learning3d.models import Classifier
from learning3d.data_utils import ClassificationData, ModelNet40Data

def _init_(args):
	if not os.path.exists('checkpoints'):
		os.makedirs('checkpoints')
	if not os.path.exists('checkpoints/' + args.exp_name):
		os.makedirs('checkpoints/' + args.exp_name)
	if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
		os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
	# os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
	# os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')

class IOStream:
	def __init__(self, path):
		self.f = open(path, 'a')

	def cprint(self, text):
		print(text)
		self.f.write(text + '\n')
		self.f.flush()

	def close(self):
		self.f.close()

def dropout_augmentation(points, drop_ratio=0.3):

    batch_size, num_points, _ = points.shape
    num_drop = int(num_points * drop_ratio)  # Number of points to remove

    # Generate random indices for each batch
    keep_indices = torch.randperm(num_points)[num_drop:]  # Select remaining points

    return points[:, keep_indices, :]  # Keep only selected points

def generate_pgd_adversarial(model, points, targets, device, eps=0.1, alpha=0.02, steps=7):
    model.eval()  # Set model to evaluation mode to prevent batch norm updates
    attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True)
    adv_points = attack(points, targets)
    model.train()  # Set model back to training mode
    return adv_points

def generate_pgd_max_loss(model, points, targets, device, eps=0.1, alpha=0.02, steps=7, k=5):
    """
    Generates multiple PGD adversarial examples and selects the one that maximizes loss.
    
    Args:
        model: The classification model.
        points: Input point cloud batch.
        targets: True labels.
        device: CUDA or CPU.
        eps: Maximum perturbation size.
        alpha: Step size.
        steps: Number of attack iterations.
        k: Number of different PGD samples to generate.

    Returns:
        The adversarial example with the highest loss.
    """
    model.eval()  # Set model to evaluation mode
    attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True)
    
    max_loss = -1  # Store the maximum observed loss
    worst_adv_points = None  # Store the most adversarial example
    
    loss_fn = torch.nn.CrossEntropyLoss()  # Loss function

    for _ in range(k):  # Generate K different PGD samples
        adv_points = attack(points, targets)  # Generate adversarial example
        outputs = model(adv_points)  # Get model predictions
        loss = loss_fn(outputs, targets)  # Compute loss

        if loss.item() > max_loss:  # Check if this adversarial example is the worst
            max_loss = loss.item()
            worst_adv_points = adv_points.clone().detach()

    model.train()  # Switch model back to training mode
    return worst_adv_points  # Return the worst adversarial example

def pgd_attack(model, test_loader, device, eps=0.1, alpha=0.02, steps=40):
	
	attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True)
	model.eval()
	total, correct = 0, 0

	for points, targets in test_loader:
		points, targets = points.to(device), targets.to(device)

		# ✅ Generate adversarial examples
		adv_points = attack(points, targets)

		# ✅ Make predictions on adversarial samples
		outputs = model(adv_points)
		_, preds = outputs.max(1)

		total += targets.size(0)
		correct += (preds == targets).sum().item()

	adv_accuracy = correct / total
	return adv_accuracy

def generate_cw_adversarial(model, points, targets, device, c=1, kappa=0, steps=1000, lr=0.01):
    model.eval()  # Switch to evaluation mode
    attack = torchattacks.CW(model, c=c, kappa=kappa, steps=steps, lr=lr)
    
    adv_points = attack(points, targets)  # Generate adversarial examples
    
    model.train()  # Switch back to training mode
    return adv_points

def cw_attack(model, test_loader, device, c=1, kappa=0, steps=1000, lr=0.01):
    attack = torchattacks.CW(model, c=c, kappa=kappa, steps=steps, lr=lr)

    model.eval()
    total, correct = 0, 0

    for points, targets in test_loader:
        points, targets = points.to(device), targets.to(device)

        # ✅ Generate adversarial examples using C&W
        adv_points = attack(points, targets)

        # ✅ Make predictions on adversarial samples
        outputs = model(adv_points)
        _, preds = outputs.max(1)

        total += targets.size(0)
        correct += (preds == targets).sum().item()

    adv_accuracy = correct / total
    return adv_accuracy

def dropout_attack(model, test_loader, device, drop_ratio=0.5):
    """Applies Dropout Attack by selectively removing important points from the point cloud."""

    model.eval()
    total, correct = 0, 0

    for points, targets in test_loader:
        points, targets = points.to(device), targets.to(device)

        num_points = points.shape[1]  # Number of points per point cloud
        num_drop = int(num_points * drop_ratio)  # Number of points to remove

        # ✅ Compute per-point variance across dimensions
        std_dev = torch.std(points, dim=0)  # Shape: [num_points, 3]
        importance = std_dev.mean(dim=-1)  # Mean importance across dimensions

        # ✅ Select the most important points to drop
        _, drop_indices = torch.topk(importance, num_drop)  # Get indices of highest-variance points
        keep_indices = torch.tensor([i for i in range(num_points) if i not in drop_indices], device=device)

        # ✅ Apply dropout
        adv_points = torch.index_select(points, 1, keep_indices)  # Keep only selected points

        # ✅ Make predictions on modified point clouds
        outputs = model(adv_points)
        _, preds = outputs.max(1)

        total += targets.size(0)
        correct += (preds == targets).sum().item()

    adv_accuracy = correct / total
    return adv_accuracy

def outlier_removal(points, threshold=1.5):
    """Removes outliers using a statistical approach based on point distances."""
    batch_size, num_points, _ = points.shape
    
    # Compute mean and standard deviation along each dimension
    mean = torch.mean(points, dim=1, keepdim=True)  # Shape: [B, 1, 3]
    std = torch.std(points, dim=1, keepdim=True)  # Shape: [B, 1, 3]
    
    # Compute z-score (distance from mean in terms of standard deviation)
    z_scores = torch.abs((points - mean) / (std + 1e-6))  # Avoid division by zero
    
    # Identify inliers (points within threshold * std deviations)
    mask = (z_scores < threshold).all(dim=-1)  # Shape: [B, N]
    
    # Keep only inlier points (replace outliers with mean value)
    filtered_points = torch.where(mask.unsqueeze(-1), points, mean)
    return filtered_points

def randomized_smoothing(points, sigma=0.05):
    """Applies Gaussian noise to each point for Randomized Smoothing defense."""
    noise = torch.normal(mean=0, std=sigma, size=points.shape, device=points.device)
    return points + noise

def test_one_epoch(device, model, test_loader):
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0
	for i, data in enumerate(tqdm(test_loader)):
		points, target = data
		# print(f"📢 Target shape before squeeze: {target.shape}")  # Debugging line
		target = target.squeeze().long()
		# print(f"📢 Target shape after squeeze: {target.shape}")  # Debugging line


		points = points.to(device)
		target = target.to(device)

		output = model(points)
		loss_val = torch.nn.functional.nll_loss(
			torch.nn.functional.log_softmax(output, dim=1), target, size_average=False)

		test_loss += loss_val.item()
		count += output.size(0)

		_, pred1 = output.max(dim=1)
		ag = (pred1 == target)
		am = ag.sum()
		pred += am.item()

	test_loss = float(test_loss)/count
	accuracy = float(pred)/count
	return test_loss, accuracy

def test(args, model, test_loader, textio, attack=None, defense=None):
	metrics_dir = os.path.expanduser("~/fyp/learning3d/src/learning3d/metrics")
	
	if attack is None:
		test_loss, test_accuracy = test_one_epoch(args.device, model, test_loader)
		textio.cprint('Validation Loss: %f & Validation Accuracy: %f'%(test_loss, test_accuracy))
	elif attack=='pgd':
		test_loss, test_accuracy = test_one_epoch(args.device, model, test_loader)  # Normal first
		textio.cprint(f"Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.4f}")
		start_time = time.time()
		if defense == 'outlier':
			filtered_data = []
			for i in range(len(test_loader.dataset.data_class)):
				points, label = test_loader.dataset.data_class[i]  # Extract points
				filtered_points = outlier_removal(points.unsqueeze(0), threshold=1.5).squeeze(0)  # Apply filtering
				filtered_data.append((filtered_points, label))  # Store modified points
    		# ✅ Replace test_loader dataset with modified version
			filtered_dataset = ClassificationData(filtered_data)
			test_loader = DataLoader(filtered_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)
			pgd_accuracy_defended = pgd_attack(model, test_loader, args.device, eps=0.1, alpha=0.02, steps=40)
			textio.cprint(f"PGD Adversarial Accuracy with Outlier Removal: {pgd_accuracy_defended:.4f}")
		elif defense=='smoothing':
			smoothed_data = []
			for i in range(len(test_loader.dataset.data_class)):
				points, label = test_loader.dataset.data_class[i]  # Extract points
				smoothed_points = randomized_smoothing(points.unsqueeze(0), sigma=0.02).squeeze(0)  # Apply noise
				smoothed_data.append((smoothed_points, label))  # Store modified points

			# ✅ Create a new DataLoader with smoothed dataset
			smoothed_dataset = ClassificationData(smoothed_data)
			test_loader = DataLoader(smoothed_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

			# ✅ Run PGD attack on smoothed dataset
			pgd_accuracy_defended = pgd_attack(model, test_loader, args.device, eps=0.1, alpha=0.02, steps=40)
			textio.cprint(f"PGD Adversarial Accuracy with Randomized Smoothing: {pgd_accuracy_defended:.4f}")
		else:
			pgd_accuracy = pgd_attack(model, test_loader, args.device, eps=0.1, alpha=0.02, steps=40)
			textio.cprint(f"PGD Adversarial Accuracy: {pgd_accuracy:.4f}")
		end_time = time.time() - start_time
		textio.cprint(f"PGD Attack Time: {end_time:.4f} seconds")
		os.makedirs(metrics_dir, exist_ok=True)
		metrics_file = os.path.join(metrics_dir, "pgd_adv_general2_dropoutaug_results.txt")
		with open(metrics_file, "w") as f:
			f.write(f"Attack Duration (PGD Attack): {end_time:.4f} seconds\n")
			if defense == 'outlier':
				f.write(f"Adversarial Accuracy (PGD Attack with Outlier Removal): {pgd_accuracy_defended:.4f}\n")	
			elif defense == 'smoothing':
				f.write(f"Adversarial Accuracy (PGD Attack with randomized smoothing): {pgd_accuracy_defended:.4f}\n")
			else:
				f.write(f"Adversarial Accuracy (PGD Attack): {pgd_accuracy:.4f}\n")

		print(f"PGD Attack evaluation completed. Results saved to {metrics_file}")
	elif attack == "cw":
		# ✅ Run C&W attack
		# cw_accuracy = cw_attack(model, test_loader, args.device, c=1, kappa=0, steps=1000, lr=0.01)
		# textio.cprint(f"CW Adversarial Accuracy: {cw_accuracy:.4f}")
		start_time = time.time()
		if defense == "outlier":
			filtered_data = []
			for i in range(len(test_loader.dataset.data_class)):
				points, label = test_loader.dataset.data_class[i]  # Extract points
				filtered_points = outlier_removal(points.unsqueeze(0), threshold=1.5).squeeze(0)  # Apply filtering
				filtered_data.append((filtered_points, label))  # Store modified points
    		# ✅ Replace test_loader dataset with modified version
			filtered_dataset = ClassificationData(filtered_data)
			test_loader = DataLoader(filtered_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)
			cw_accuracy_defended = cw_attack(model, test_loader, args.device, c=1, kappa=0, steps=1000, lr=0.01)
			textio.cprint(f"CW Adversarial Accuracy with Outlier Removal: {cw_accuracy_defended:.4f}")
		elif defense=='smoothing':
			smoothed_data = []
			for i in range(len(test_loader.dataset.data_class)):
				points, label = test_loader.dataset.data_class[i]  # Extract points
				smoothed_points = randomized_smoothing(points.unsqueeze(0), sigma=0.02).squeeze(0)  # Apply noise
				smoothed_data.append((smoothed_points, label))  # Store modified points

			# ✅ Create a new DataLoader with smoothed dataset
			smoothed_dataset = ClassificationData(smoothed_data)
			test_loader = DataLoader(smoothed_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

			# ✅ Run CW attack on smoothed dataset
			cw_accuracy_defended =  cw_attack(model, test_loader, args.device, c=1, kappa=0, steps=1000, lr=0.01)
			textio.cprint(f"PGD Adversarial Accuracy with Randomized Smoothing: {cw_accuracy_defended:.4f}")
		else:
			cw_accuracy = cw_attack(model, test_loader, args.device, c=1, kappa=0, steps=1000, lr=0.01)
			textio.cprint(f"CW Adversarial Accuracy: {cw_accuracy:.4f}")
		end_time = time.time() - start_time
		# ✅ Save C&W attack results
		metrics_file = os.path.join(metrics_dir, "cw_adv_general2_dropoutaug_results.txt")
		with open(metrics_file, "w") as f:
			f.write(f"Attack Duration: {end_time:.4f} seconds\n")
			if defense == "outlier":
				f.write(f"Adversarial Accuracy (C&W Attack with Outlier Defence): {cw_accuracy_defended:.4f}\n")
			elif defense == 'smoothing':
				f.write(f"Adversarial Accuracy (C&W Attack with Randomized Smoothing Defence): {cw_accuracy_defended:.4f}\n")
			else:
				f.write(f"Adversarial Accuracy (C&W Attack): {cw_accuracy:.4f}\n")

		print(f"C&W Attack evaluation completed. Results saved to {metrics_file}")
	elif attack == "dropout":
		# ✅ Run Dropout Attack
		start_time = time.time()
		if defense == "outlier":
			filtered_data = []
			for i in range(len(test_loader.dataset.data_class)):
				points, label = test_loader.dataset.data_class[i]  # Extract points
				filtered_points = outlier_removal(points.unsqueeze(0), threshold=1.5).squeeze(0)  # Apply filtering
				filtered_data.append((filtered_points, label))  # Store modified points
    		# ✅ Replace test_loader dataset with modified version
			filtered_dataset = ClassificationData(filtered_data)
			test_loader = DataLoader(filtered_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)
			drop_accuracy_defended = dropout_attack(model, test_loader, args.device, drop_ratio=0.3)
			textio.cprint(f"Dropout Adversarial Accuracy with Outlier Removal: {drop_accuracy_defended:.4f}")
		elif defense == 'smoothing':
			smoothed_data = []
			for i in range(len(test_loader.dataset.data_class)):
				points, label = test_loader.dataset.data_class[i]  # Extract points
				smoothed_points = randomized_smoothing(points.unsqueeze(0), sigma=0.02).squeeze(0)  # Apply noise
				smoothed_data.append((smoothed_points, label))  # Store modified points

			# ✅ Create a new DataLoader with smoothed dataset
			smoothed_dataset = ClassificationData(smoothed_data)
			test_loader = DataLoader(smoothed_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)
			drop_accuracy_defended = dropout_attack(model, test_loader, args.device, drop_ratio=0.3)
			textio.cprint(f"Dropout Adversarial Accuracy with Randomized smoothing: {drop_accuracy_defended:.4f}")
		else:
			drop_accuracy = dropout_attack(model, test_loader, args.device, drop_ratio=0.3)
			textio.cprint(f"Dropout Adversarial Accuracy: {drop_accuracy:.4f}")
		end_time = time.time() - start_time
		# ✅ Save Dropout attack results
		metrics_file = os.path.join(metrics_dir, "dropout_adv_general2_dropoutaug_results.txt")
		with open(metrics_file, "w") as f:
			f.write(f"Attack Duration: {end_time:.4f} seconds\n")
			if defense == "outlier":	
				f.write(f"Adversarial Accuracy (Dropout Attack with Outlier Removal): {drop_accuracy_defended:.4f}\n")
			elif defense == "smoothing":	
				f.write(f"Adversarial Accuracy (Dropout Attack with Randomized Smoothing Defence): {drop_accuracy_defended:.4f}\n")
			else:
				f.write(f"Adversarial Accuracy (Dropout Attack): {drop_accuracy:.4f}\n")

		print(f"✅ Dropout Attack evaluation completed. Results saved to {metrics_file}")

def train_one_epoch(device, model, train_loader, optimizer, adv_training=False, adv_type="pgd", defense=None):
	model.train()
	train_loss = 0.0
	pred  = 0.0
	count = 0
	# start_time = time.time()
	for i, data in enumerate(tqdm(train_loader)):
		points, target = data
		target = target.squeeze().long()
		
		num_classes = 33  
		if torch.any(target < 0) or torch.any(target >= num_classes):
			print(f"❌ ERROR: Found invalid labels in batch {i}! Unique labels: {torch.unique(target)}")
			exit(1)  # Stop execution to debug the issue

		points = points.to(device)
		target = target.to(device)

		if defense == "dropout_aug":
			points = dropout_augmentation(points, drop_ratio=0.3)

		if adv_training and adv_type == "pgd":
			# adv_points = generate_pgd_adversarial(model, points, target, device)
			adv_points = generate_pgd_max_loss(model, points, target, device, eps=0.2, alpha=0.04, steps=100, k=3)
			adv_targets = target.clone()
		elif adv_training and adv_type == "cw":
			adv_points = generate_cw_adversarial(model, points, target, device, c=1, kappa=0, steps=1000, lr=0.01)
			adv_targets = target.clone()
		elif adv_training and adv_type=="general":
			adv_pgd = generate_pgd_adversarial(model, points, target, device, eps=0.2, alpha=0.04, steps=100)
			adv_cw = generate_cw_adversarial(model, points, target, device, c=1, kappa=0, steps=1000, lr=0.01)
			adv_points = torch.cat([adv_pgd, adv_cw], dim=0)  # Combine both attacks
			adv_targets = torch.cat([target, target], dim=0)  # Ensure matching labels
		else:
			adv_points = points.clone()
			adv_targets = target.clone()

		output_clean = model(points)
		output_adv = model(adv_points)
		loss_val_clean = torch.nn.functional.nll_loss(
			torch.nn.functional.log_softmax(output_clean, dim=1), target, size_average=False)
		loss_val_adv = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(output_adv, dim=1), adv_targets, size_average=False)
		loss_val = 0.5 * (loss_val_clean + loss_val_adv) if adv_training else loss_val_clean
	
		optimizer.zero_grad()
		loss_val.backward()
		optimizer.step()

		train_loss += loss_val.item()
		count += target.size(0)

		_, pred1 = output_clean.max(dim=1)
		ag = (pred1 == target)
		am = ag.sum()
		pred += am.item()

	# train_time = time.time() - start_time
	train_loss = float(train_loss)/count
	accuracy = float(pred)/count
	return train_loss, accuracy

def train(args, model, train_loader, test_loader, boardio, textio, checkpoint):
	learnable_params = filter(lambda p: p.requires_grad, model.parameters())
	if args.optimizer == 'Adam':
		optimizer = torch.optim.Adam(learnable_params, lr=0.001)  # Adjust learning rate
	else:
		optimizer = torch.optim.SGD(learnable_params, lr=0.1, momentum=0.9)  # Added momentum

	# ✅ Add Learning Rate Scheduler
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

	if checkpoint is not None:
		min_loss = checkpoint['min_loss']
		optimizer.load_state_dict(checkpoint['optimizer'])

	best_test_loss = np.inf
	best_test_accuracy = 0.0  # ✅ Track best accuracy
	patience = 10
	counter = 0  

	# ✅ Define separate save directories
	pretrained_dir = os.path.expanduser("~/fyp/learning3d/src/learning3d/pretrained")  # For models
	metrics_dir = os.path.expanduser("~/fyp/learning3d/src/learning3d/metrics")   # For logs

	# ✅ Ensure directories exist
	os.makedirs(pretrained_dir, exist_ok=True)
	os.makedirs(metrics_dir, exist_ok=True)

	# metrics_file = os.path.join(metrics_dir, "training_metrics.txt")  # ✅ Define metrics file path
	if args.adv_training and args.adv_type == "pgd":
		if args.defense == "dropout_aug":
			model_name = 'best_adv_pgdmax_dropoutaug_model.t7'
			metrics_file = os.path.join(metrics_dir, "adv_pgdmax_dropoutaug_training_metrics.txt")
		else:
			# model_name = 'best_adv_model.t7'
			model_name = 'best_adv_pgd_model.t7'
			# metrics_file = os.path.join(metrics_dir, "adv_training_metrics.txt")
			metrics_file = os.path.join(metrics_dir, "adv_max_training_metrics.txt")
	elif args.adv_training and args.adv_type == "cw":
		model_name = 'best_adv_cw_model.t7'
		metrics_file = os.path.join(metrics_dir, "adv_cw_training_metrics.txt")
	elif args.adv_training and args.adv_type=="general":
		if args.defense == "dropout_aug":
			model_name = 'best_adv_general_dropoutaug_model.t7'
			metrics_file = os.path.join(metrics_dir, "adv_general_dropoutaug_training_metrics.txt")
		else:
			model_name = 'best_adv_general_model.t7'
			metrics_file = os.path.join(metrics_dir, "adv_general_training_metrics.txt")
	elif not args.adv_training and args.defense== "dropout_aug":
		model_name = 'best_base_dropoutaug_model.t7'
		metrics_file = os.path.join(metrics_dir, "base_dropoutaug_training_metrics.txt")
	else:
		model_name = 'best_base_model.t7'
		metrics_file = os.path.join(metrics_dir, "training_metrics.txt")

	# ✅ Start tracking overall training time
	start_time = time.time()  

	for epoch in range(args.start_epoch, args.epochs):
		train_loss, train_accuracy = train_one_epoch(args.device, model, train_loader, optimizer, adv_training=args.adv_training, adv_type=args.adv_type, defense=args.defense)
		test_loss, test_accuracy = test_one_epoch(args.device, model, test_loader)
		
		# ✅ Save only the best model based on test loss in `pretrained/`
		if test_loss < best_test_loss:
			best_test_loss = test_loss
			best_test_accuracy = test_accuracy  # ✅ Store corresponding accuracy
			best_epoch = epoch + 1
			counter = 0
			torch.save(model.state_dict(), os.path.join(pretrained_dir, model_name))
			# print(f"✅ Model Updated at Epoch {epoch+1} (Best Loss: {best_test_loss:.4f}, Best Accuracy: {best_test_accuracy:.4f})")
			print(f"✅ Model Updated at Epoch {epoch+1} ({'Adversarial' if args.adv_training else 'Normal'} Training)")
		else:
			counter += 1
			if counter >= patience:
				print(f"Early stopping implemented at {epoch+1}.")
				break
		scheduler.step()
		
		boardio.add_scalar('Train Loss', train_loss, epoch+1)
		boardio.add_scalar('Test Loss', test_loss, epoch+1)
		boardio.add_scalar('Best Test Loss', best_test_loss, epoch+1)
		boardio.add_scalar('Train Accuracy', train_accuracy, epoch+1)
		boardio.add_scalar('Test Accuracy', test_accuracy, epoch+1)
		
		textio.cprint(f"EPOCH:: {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Best Loss: {best_test_loss:.4f}")
		textio.cprint(f"EPOCH:: {epoch+1}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

	# ✅ Stop the timer after all epochs are completed
	total_training_time = time.time() - start_time  

	# ✅ Save final metrics in `metrics_logs/` 
	with open(metrics_file, "w") as f:  # Open in write mode (overwrite if exists)
		f.write(f"Total Training Time: {total_training_time:.2f} seconds\n")
		f.write("="*50 + "\n")
		f.write(f"Best Test Loss: {best_test_loss:.4f}\n")
		f.write(f"Best Test Accuracy: {best_test_accuracy:.4f}\n")  # ✅ Save Best Accuracy
		f.write(f"Best Epoch: {best_epoch+1}\n")

	print(f"✅ Training Completed! Total Time: {total_training_time:.2f} seconds, Best Accuracy: {best_test_accuracy:.4f}")

def options():
	parser = argparse.ArgumentParser(description='Point Cloud Registration')
	parser.add_argument('--exp_name', type=str, default='exp_classifier', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--dataset_path', type=str, default='ModelNet40',
						metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
	parser.add_argument('--adv_training', action='store_true', help='Enable adversarial training using PGD examples')
	parser.add_argument('--adv_type', type=str, default='pgd', choices=['pgd', 'cw', 'general'],
                    help='Type of adversarial training to use (pgd or cw).')
	parser.add_argument('--eval', action='store_true', help='Evaluate the model (test mode)')
	parser.add_argument('--attack', type=str, default=None, choices=[None, "pgd", "cw", "dropout"],
                        help='Type of adversarial attack (e.g., "pgd"). If not set, runs normal test.')
	parser.add_argument('--defense', type=str, default=None, choices=['outlier', 'smoothing', 'dropout_aug'],
                    help='Defense method to apply during evaluation (e.g., outlier)')

	# settings for input data
	parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
						metavar='DATASET', help='dataset type (default: modelnet)')
	parser.add_argument('--num_points', default=1024, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')
	parser.add_argument('--root_dir', default='./', type=str, 
					 	help='path of the data where modelnet files are downloaded.')

	# settings for PointNet
	parser.add_argument('--pointnet', default='tune', type=str, choices=['fixed', 'tune'],
						help='train pointnet (default: tune)')
	parser.add_argument('--emb_dims', default=1024, type=int,
						metavar='K', help='dim. of the feature vector (default: 1024)')
	parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
						help='symmetric function (default: max)')

	# settings for on training
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('-j', '--workers', default=4, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--batch_size', default=32, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--epochs', default=200, type=int,
						metavar='N', help='number of total epochs to run')
	parser.add_argument('--start_epoch', default=0, type=int,
						metavar='N', help='manual epoch number (useful on restarts)')
	parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
						metavar='METHOD', help='name of an optimizer (default: Adam)')
	parser.add_argument('--resume', default='', type=str,
						metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
	parser.add_argument('--pretrained', default='', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')

	args = parser.parse_args()
	return args

def main():
	args = options()
	args.dataset_path = os.path.expanduser("~/fyp/finaldataset")
	args.root_dir = args.dataset_path

	torch.backends.cudnn.deterministic = True
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)

	boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
	_init_(args)

	textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
	textio.cprint(str(args))

	
	trainset = ClassificationData(ModelNet40Data(train=True, root_dir=args.root_dir, use_normals=False))
	testset = ClassificationData(ModelNet40Data(train=False, root_dir=args.root_dir, use_normals=False))
	train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
	test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	# Create PointNet Model.
	num_classes = 33
	ptnet = PointNet(emb_dims=args.emb_dims, use_bn=True)
	model = Classifier(feature_model=ptnet, num_classes=num_classes)

	checkpoint = None
	if args.resume:
		assert os.path.isfile(args.resume)
		checkpoint = torch.load(args.resume)
		args.start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['model'])

	if args.pretrained:
		assert os.path.isfile(args.pretrained)
		model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
	model.to(args.device)

	if args.eval:
		test(args, model, test_loader, textio, attack=args.attack, defense=args.defense)
	else:
		train(args, model, train_loader, test_loader, boardio, textio, checkpoint)

if __name__ == '__main__':
	main()