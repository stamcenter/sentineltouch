import torch
from tqdm import tqdm
from datetime import datetime
import os
import csv


def train_model(model, train_dataloader, arcface, criterion, optimizer, val_loader=None, scheduler=None, num_epochs=10, device: torch.device = torch.device('cpu'), model_name=datetime.now().strftime('%Y-%m-%d')):
	for epoch in range(num_epochs):
		
		model.train()
		total_loss = 0
		total = 0

		# print("Thresholds [0.0 | 0.2 | 0.4 | 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:")
		loop = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=True)
		for images, labels in loop:
			images, labels = images.to(device), labels.long().to(device)
			optimizer.zero_grad()
			logits = model(images)
			logits = arcface(logits, labels)

			loss = criterion(logits, labels)

			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			total += labels.size(0)


			loop.set_description_str(f"Epoch [{epoch+1}/{num_epochs}]")
			loop.set_postfix({
				'loss': f'{total_loss / total :.8f}'
			})
		if val_loader is not None:
			evaluate_1n_model(model, val_loader, device)

		if scheduler:
			scheduler.step()

	print("\nTraining Summary:")
	print(f"  Loss: {total_loss / len(train_dataloader.dataset):.4f}")
	# print(f"  AUC Score:  {auc * 100:.4f}")

	torch.save(model.state_dict(), f"./../trained_models/sentinal_1_to_N_{model_name}.pth")

def evaluate_1n_model(model, test_loader, device="cpu", mode = "threshold"):
	model.eval()
	embeddings = []
	labels = []

	# Pass 1: Build the embedding database (augmented view 1)
	with torch.no_grad():
		for i, (inputs, label) in enumerate(test_loader):
			x = inputs.to(device)  # inputs is a tuple: (x1, x2)
			label = label.cpu()

			emb = model.get_embedding(x).cpu()  # shape: [1, D]
			# print(f"{i}: emb len:{len(emb)} -  {emb}")
			embeddings.append(emb)
			labels.append(label)

	embeddings = torch.cat(embeddings, dim=0)  # shape: [N, D]

	# Compute similarity matrix. Since model.embedding produces L2 normalized vectors,
	# this is equivalent to cosine similarity.
	mat_similarity = embeddings.matmul(embeddings.T)  # shape: [N, N]
	# check if every element is nan
	if torch.isnan(mat_similarity).all():
		print("NaN values found in similarity matrix")

	labels = torch.cat(labels, dim=0)  # shape: [N]

	labels = labels.view(-1, labels.size(0)) == labels.view(labels.size(0), -1)

	accuracy = []

	total_comp = torch.ones_like(mat_similarity).triu(1)
	total_comp = total_comp.sum().item()

	thresh_best_acc = 0
	# for threshold in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
	for threshold in [0.95]:
		threshed = mat_similarity > threshold
		
		#remove diagonal
		correct = (threshed == labels).triu(1).sum()

		accuracy.append(correct / total_comp)
		if accuracy[-1] > thresh_best_acc:
			thresh_best_acc = accuracy[-1]
	
	tqdm.write(f'Validation Accuracy: {" | ".join(f"{acc * 100:.03f}" for acc in accuracy)}')

def save_embeddings(model, dataloader, device, save_dir, dataset_name, embedding_space, model_type, task):
	model.eval()

	dir_name = os.path.join(save_dir, dataset_name, model_type, str(embedding_space), task)
	os.makedirs(dir_name, exist_ok=True)

	# Pass 1: Build the embedding database (augmented view 1)
	with torch.no_grad():
		for i, (inputs, label) in enumerate(dataloader):
			x = inputs.to(device)
			label = label.cpu()

			emb = model.get_embedding(x).cpu()  # shape: [1, D]
			emb = emb.numpy().flatten()
			# save as csv
			with open(os.path.join(dir_name, f"{i}.csv"), mode='w', newline='') as f:
				writer = csv.writer(f)
				writer.writerow(emb)