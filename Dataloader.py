from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
import torch
class Summarization_Dataset(Dataset):
    def __init__(self, dataset, doc_max_timesteps, sent_encoder_type='bert', summary_type='oracle', summary_level='sen'):
        super(Summarization_Dataset, self).__init__()
        self.dataset = dataset
        self.doc_max_timesteps = doc_max_timesteps
        self.length = len(self.dataset['data'])
        self.summary_type = summary_type
        self.summary_level = summary_level
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

        self.sent_encoder_type = sent_encoder_type
        if self.sent_encoder_type == 'sbert':
            self.sen_encoder = SentenceTransformer('all-mpnet-base-v2', device='cuda')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.sen_encoder = BertModel.from_pretrained("bert-base-uncased")
            self.sen_encoder.to(self.device)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():
            article = self.dataset['data'][idx]['text']
            sen_labels = self.dataset['data'][idx]['label']
            try:
                summary = self.dataset['data'][idx]['summary']
            except:
                summary = ''
            article_sens = article
            label_shape = (len(article), len(sen_labels))  # [N, len(label)]
            label_matrix = np.zeros(label_shape, dtype=int)
            if sen_labels != []:
                label_matrix[np.array(sen_labels), np.arange(len(sen_labels))] = 1

            label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
            N, m = label_m.shape
            if m < self.doc_max_timesteps:
                pad_m = np.zeros((N, self.doc_max_timesteps - m))
                label_m = np.hstack([label_m, pad_m])

            if self.sent_encoder_type == 'sbert':
                article_sen_embeds = self.sen_encoder.encode(article_sens, convert_to_tensor=True).cpu()

            elif self.sent_encoder_type == 'bert':
                inputs = self.tokenizer(article_sens, return_tensors="pt", padding=True, truncation=True, max_length=500).to(self.device)
                article_sen_embeds = self.sen_encoder(**inputs).last_hidden_state[:, 0, :] #(sen_num, word_num, emb_dim)

            return [], {'article_sen_embeds': article_sen_embeds,
                        'article_sens': article_sens,
                        'label_m': label_m,
                        'label':sen_labels,
                        'summary':summary
                        }
def load_data_summarization(batch_size, doc_max_timesteps=50, data_name='cnn',
                            sent_encoder_type='bert', summary_type='oracle', summary_level='sen'):
    def collect_fn(data):
        article_sen_embeds = [d[1]['article_sen_embeds'] for d in data]
        label_m = torch.tensor([np.pad((d[1]['label_m'].sum(-1)), (0, doc_max_timesteps-len((d[1]['label_m'].sum(-1)))), mode='constant', constant_values=0) for d in data],dtype=torch.int64)
        article_sen_masks = torch.zeros(len(article_sen_embeds), doc_max_timesteps)
        article_sen_embeds_padded = torch.zeros(
            (len(article_sen_embeds), doc_max_timesteps, article_sen_embeds[0].shape[1]))
        for i, embs in enumerate(article_sen_embeds):
            if embs.shape[0] > doc_max_timesteps:
                article_sen_embeds_padded[i, :, :] = embs[:doc_max_timesteps, :]
                article_sen_masks[i, :] = 1
            else:
                article_sen_embeds_padded[i, 0:embs.shape[0], :] = embs
                article_sen_masks[i, 0:embs.shape[0]] = 1
        return {'article_sen_embeds': article_sen_embeds_padded, 'article_sen_masks': article_sen_masks,
                'label': label_m}

    if data_name == 'cnn':
        train_data = load_dataset('json', data_files={'data': ['data/CNNDM/train.label.jsonl']})
        eval_data = load_dataset('json', data_files={'data': ['data/CNNDM/val.label.jsonl']})
        test_data = load_dataset('json', data_files={'data': ['data/CNNDM/test.label.jsonl']})
    trainDateset = Summarization_Dataset(dataset=train_data, doc_max_timesteps=doc_max_timesteps, sent_encoder_type=sent_encoder_type, summary_type=summary_type,
                               summary_level=summary_level)
    evalDateset = Summarization_Dataset(dataset=eval_data, doc_max_timesteps=doc_max_timesteps,
                                         sent_encoder_type=sent_encoder_type, summary_type=summary_type,
                                         summary_level=summary_level)
    testDateset = Summarization_Dataset(dataset=test_data, doc_max_timesteps=doc_max_timesteps,
                                         sent_encoder_type=sent_encoder_type, summary_type=summary_type,
                                         summary_level=summary_level)
    train_dataloader = DataLoader(
        trainDateset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collect_fn
    )
    eval_dataloader = DataLoader(
        evalDateset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collect_fn
    )
    test_dataloader = DataLoader(
        testDateset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collect_fn,
    )
    return train_dataloader, (eval_dataloader,evalDateset),(test_dataloader,testDateset)

