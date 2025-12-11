from sklearn.metrics import auc, roc_curve
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import numpy as np

from scipy.optimize import brentq
from scipy.interpolate import interp1d

def train_model(model, train_dataloader, contrastive_loss, optimizer, val_loader=None, scheduler=None, num_epochs=10, device: torch.device = torch.device('cpu'), model_name=datetime.now().strftime('%Y-%m-%d')):
	for epoch in range(num_epochs):
		
		model.train()
		total_loss = 0
		total = 0

		loop = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=True)
		
		all_labels = []
		all_logits = []

		for x1, x2, labels in loop:
			x1, x2, labels = x1.to(device), x2.to(device), labels.long().to(device)
			feat1 = model.get_embedding(x1)
			feat2 = model.get_embedding(x2)

			logits = F.cosine_similarity(feat1, feat2)
			loss = contrastive_loss(logits, labels)

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			total_loss += loss.item()
			total += labels.size(0)

			all_labels.extend(labels.cpu().numpy())
			all_logits.extend(logits.cpu().detach().numpy())

			loop.set_description_str(f"Epoch [{epoch+1}/{num_epochs}]")
			loop.set_postfix({
				'loss': f'{total_loss / total :.8f}'
			})
                  
		thresholds, th, auc_score, eer, acc, metrics, FRR, TPR = auc_loop(all_logits, all_labels)
		tqdm.write(f"TRAIN--LOSS: {total_loss / total:.04f} AUC: {auc_score:.04f}, EER: {eer:.04f}, Acc: {acc:.04f}, Precision: {metrics[0]:.04f}, Recall: {metrics[1][0]:.04f}, F1s: {metrics[2][0]:.04f} Threshold: {th}")
		if val_loader is not None:
			evaluate_1n_model(model, val_loader, contrastive_loss, device)

		if scheduler:
			scheduler.step()

	torch.save(model.state_dict(), f"./../trained_models/blind_match_1_to_1_{model_name}.pth")

def eval_state(probs, labels, thr):
    labels = np.array(labels)
    probs = np.array(probs)

    predict = probs >= thr
    labels = np.array(labels)
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP


def auc_loop(scores, labels, pos_label=1.0):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=pos_label)
    # print(thresholds)
    th, acc, metrics = find_best_threshold(scores, labels, thresholds=thresholds, pos_label=pos_label)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    auc_score = auc(fpr, tpr)
    TN, FN, FP, TP = eval_state(scores, labels, th)
    TPR = TP / float(TP + FN + 1e-8)
    FRR = FN / float(FN + TP + 1e-8)
    # print(f"At threshold = {th}", TN, FN, FP, TP, len(labels))
    return thresholds, th, auc_score, eer, acc, metrics, FRR, TPR

# 1-FPIR ~= FA = FM  = False Match     - same but predicted as diff
# FNIR ~= FR = FNM = False Non-match - diff but predicted as same
def eer_from_fpr_tpr(fpr, tpr, thresholds):
    '''
    Equal-Error Rate (EER)  : it denotes the error rate at the threshold t 
                            for which both false match rate and false non-match rate are identical
                            : FMR(t) = FNMR(t)
                            
    Although EER is an important indicator, a fingerprint system is
    rarely used at the operating point corresponding to EER (because the corresponding
    FMR is often not sufficiently low for security requirements), and often a more stringent
    threshold is set corresponding to a pre-specified value of FMR.
    =================================================================From FVC2004 paper.
    '''
    eer = 1.0 
    for i in range(len(thresholds)):
        if fpr[i] >= 1 - tpr[i]:
            eer = min(eer, (fpr[i] + (1 - tpr[i])) / 2)
    return eer

def find_best_threshold(scores, labels, thresholds, pos_label=1):
    labels = np.array(labels)
    scores = np.array(scores)
    
    best_acc, best_ms = 0, [0,0,0]
    best_thresh = None
    for i, thresh in enumerate(thresholds):
        # print(thresh)
        # compute accuracy for this threshold
        pred_labels = [1 if s >= thresh else 0 for s in scores]
        acc = sum([1 if pred_labels[j] == labels[j] else 0 for j in range(len(labels))]) / len(labels)
        true_positives = sum([1 if (pred_labels[j] == labels[j])&(labels[j] == pos_label) else 0 for j in range(len(labels))])
        predicted_positives = sum([pos_label if pred_labels[j] == labels[j] else 1-pos_label for j in range(len(labels))])
        possible_positives = sum(labels == pos_label)
        precision = true_positives / (predicted_positives + 1e-8)  
        recall = true_positives / (possible_positives + 1e-8)      
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)  

        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            best_ms = [precision, recall, f1_score]

    return best_thresh, best_acc, best_ms

def evaluate_1n_model(model, test_loader, contrastive_loss, device="cpu", mode = "threshold"):
	model.eval()
	total_loss = 0
	total = 0

	all_labels = []
	all_logits = []

	for x1, x2, labels in test_loader:
		x1, x2, labels = x1.to(device), x2.to(device), labels.long().to(device)
		feat1 = model.get_embedding(x1)
		feat2 = model.get_embedding(x2)

		logits = F.cosine_similarity(feat1, feat2)
		loss = contrastive_loss(logits, labels)

		total_loss += loss.item()
		total += labels.size(0)

		all_labels.extend(labels.cpu().numpy())
		all_logits.extend(logits.cpu().detach().numpy())
            
	thresholds, th, auc_score, eer, acc, metrics, FRR, TPR = auc_loop(all_logits, all_labels)
	tqdm.write(f"EVAL---LOSS: {total_loss / total:.04f} AUC: {auc_score:.04f}, EER: {eer:.04f}, Acc: {acc:.04f}, Precision: {metrics[0]:.04f}, Recall: {metrics[1][0]:.04f}, F1s: {metrics[2][0]:.04f} Threshold: {th} \n\n")