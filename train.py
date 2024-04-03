from Dataloader import load_data_summarization
from model import classifier
import time
import os
import torch
from tqdm import tqdm
import random
import numpy as np
from tensorboardX import SummaryWriter
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
    parser.add_argument('--log_path', type=str, default='log/', help='path for saving the log')
    parser.add_argument('--save_path', type=str, default='save/', help='path for saving the final model')
    parser.add_argument('--gpu', type=str, default='3', help='GPU ID to use. [default: 0]')
    parser.add_argument('--doc_max_timesteps', type=int, default=50,help='max length of documents (max timesteps of documents)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.set_printoptions(threshold=50000)

    writer = SummaryWriter(args.log_path)

    Classifier = classifier()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    Classifier = Classifier.to(device=device)

    train_loader, (eval_loader, eval_dataset), _ = load_data_summarization(batch_size=args.batch_size)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Classifier.parameters()), lr=args.lr,
                                 betas=(0.9, 0.999))
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    best_Loss = 1e5
    for epoch in tqdm(range(args.epochs)):
        train_loss = 0.0
        epoch_loss = 0.0
        epoch_start_time = time.time()
        for i_batch, sample_batched in enumerate(train_loader):
            Classifier.train()
            iter_start_time = time.time()
            article_sen_embeds = sample_batched['article_sen_embeds']
            label = sample_batched['label']
            article_sen_embeds = article_sen_embeds.to(device=device)
            label = label.to(device=device)
            outputs = Classifier.forward(article_sen_embeds)
            outputs = outputs.view(-1, 2)
            target = label.view(-1)
            loss = criterion(outputs, target)
            loss = loss.reshape(-1, args.doc_max_timesteps).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Classifier.parameters(), args.max_grad_norm)
            optimizer.step()
            train_loss += float(loss.data)
            epoch_loss += float(loss.data)
            if i_batch % 100 == 0:
                print('       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | '
                      .format(i_batch, (time.time() - iter_start_time), float(train_loss / 100)))
                writer.add_scalar('training_loss', float(train_loss / 100), i_batch // 100)
                train_loss = 0.0
        epoch_avg_loss = epoch_loss / len(train_loader)
        logger.info('   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | '
                    .format(epoch, (time.time() - epoch_start_time), float(epoch_avg_loss)))

        print("Starting eval for this model ...")
        Classifier.eval()
        target_label = []
        eval_loss = 0
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(eval_loader):
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
                loss = loss.reshape(-1, args.doc_max_timesteps).sum(-1).mean()
                eval_loss += float(loss.data)

        eval_loss = eval_loss / len(eval_loader)
        if eval_loss < best_Loss:
            torch.save(Classifier.state_dict(), f'{args.save_path}/best_model.pth')
            best_Loss = eval_loss
        print("Eval loss: ", eval_loss)
        target_label = torch.vstack(target_label)
        hyps = []
        refs = []
        for index, article in enumerate(tqdm(eval_dataset)):
            targets = torch.nonzero(target_label[index] == 1).squeeze()
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)
            hyp = "\n".join(article[1]['article_sens'][target] for target in targets if target.item() < len(article[1]['article_sens']))
            hyps.append(hyp)
            ref = "\n".join(article[1]['summary'])
            refs.append(ref)
        scores_all = rouge_corpus(refs, hyps)
        res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
            scores_all['rouge1']['precision'], scores_all['rouge1']['recall'], scores_all['rouge1']['fmeasure']) \
              + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
                  scores_all['rouge2']['precision'], scores_all['rouge2']['recall'], scores_all['rouge2']['fmeasure']) \
              + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
                  scores_all['rougeLsum']['precision'], scores_all['rougeLsum']['recall'],
                  scores_all['rougeLsum']['fmeasure'])
        print(res)
    writer.close()
