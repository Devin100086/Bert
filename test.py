from Dataloader import load_data_summarization
from model import classifier
import time
import os
import torch
from tqdm import tqdm
import random
import numpy as np
import argparse
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bert Model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training/testing')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='for gradient clipping max gradient normalization')
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument('--save_path', type=str, default='save/', help='path for saving the final model')
    parser.add_argument('--gpu', type=str, default='3', help='GPU ID to use. [default: 0]')
    args = parser.parse_args()

    _, _, (test_loader, test_dataset) = load_data_summarization(batch_size=args.batch_size)
    Classifier = classifier()
    # Classifier.load_state_dict(torch.load(f'{args.save_path}/bast_model.pth'))
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    Classifier = Classifier.to(device=device)
    print("Starting test for this model ...")
    target_label = []
    test_loss = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(tqdm(test_loader)):
            article_sen_embeds = sample_batched['article_sen_embeds']
            target = sample_batched['label']
            target = target.to(device=device)
            article_sen_embeds = article_sen_embeds.to(device=device)
            outputs = Classifier.forward(article_sen_embeds)
            _, label = torch.max(outputs, dim=2)
            target_label.append(label.cpu())
            outputs = outputs.view(-1, 2)
            target = target.view(-1)
            loss = criterion(outputs, target)
            loss = loss.reshape(args.batch_size, -1).sum(-1).mean()
            test_loss += float(loss.data)

    test_loss = test_loss / len(eval_loader)
    print("Test loss: ", eval_loss)
    target_label = torch.vstack(target_label)
    hyps = []
    refs = []
    for index, article in enumerate(test_dataset):
        targets = torch.nonzero(target_label[index] == 1).squeeze()
        hyp = "\n".join(article['article_sens'][target] for target in targets)
        hyps.append(hyp)
        refs.append(article['summary'])
    scores_all = rouge_corpus(refs, hyps)
    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
        scores_all['rouge1']['precision'], scores_all['rouge1']['recall'], scores_all['rouge1']['fmeasure']) \
          + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
              scores_all['rouge2']['precision'], scores_all['rouge2']['recall'], scores_all['rouge2']['fmeasure']) \
          + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
              scores_all['rougeLsum']['precision'], scores_all['rougeLsum']['recall'],
              scores_all['rougeLsum']['fmeasure'])
    print(res)
