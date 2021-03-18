import time
from barbar import Bar
import torch
import numpy as np
from nfl.dataset import *
from tqdm import tqdm
import copy
import pandas as pd
from cnn3d.metric import evaluate_df
from cnn3d.postprocess import keep_maximums
from sklearn.metrics import f1_score
from scipy.special import softmax

class ImpactF1:
    def __init__(self, num_propagate_impacts=0):
        df_path = "train_folds_propagate_0.csv"
        df = pd.read_csv(df_path)
        gtdf = df.query("impact == 1 and visibility > 0 and confidence > 1")
        self.gtdf = gtdf
        self.df = df

    def evaluate(self, unique_ids, predictions, logits):
        scores = softmax(logits, axis=1)[:, 1]
        #np.save('scores', scores)
        preddf = self.df.copy()
        preddf['impact'] = 0
        preddf.loc[unique_ids, 'scores'] = scores
        best_f1 = 0
        best_eval = [0, 0, 0]
        best_thresh = -1
        min_thresh = 0.1
        for thresh in np.arange(min_thresh, 0.9, 0.1):
            predictions = (scores > thresh).astype(int)
            preddf.loc[unique_ids, 'impact'] = predictions
            preddf_ = preddf[preddf['impact'] == 1].copy()
            if thresh == min_thresh:
                preddf_.to_csv('cnn3d/output/predictions.csv')
            preddf__ = keep_maximums(preddf_, iou_thresh=0.25, dist=2)
            prec, rec, f1 = evaluate_df(self.gtdf, preddf__, impact=True)
            if f1 > best_f1 and len(preddf['video'].unique() == len(preddf['video'].unique())):
                best_eval = (prec, rec, f1)
                best_thresh = thresh
                best_f1 = f1
        return best_eval, best_thresh


class ThreshF1:
    def __init__(self):
        pass

    def evaluate(self, targets, logits):
        scores = softmax(logits, axis=1)[:, 1]
        best_f1 = 0
        best_thresh = -1
        for thresh in np.arange(0, 1, 0.1):
            predictions =(scores > thresh).astype(int)
            f1 = f1_score(np.array(targets).astype(int), predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return best_f1, best_thresh


def test(dataloader, model, criterion, impact_f1_metric, thresh_f1_metric, device):
    start_time = time.time()
    model.eval()
    test_loss = 0
    tp, preds, targs = 0, 0, 0
    n_batches = 0
    unique_ids = []
    predictions = []
    scores = []
    targets = []
    with torch.no_grad():
        for item in Bar(dataloader):
            data, impact = item[IMAGE_KEY].to(device), item[LABELS_KEY].to(device)
            cur_unique_ids = item[INDEX_KEY].numpy()
            unique_ids.extend(cur_unique_ids)

            output = model(data)
            loss = criterion(output, impact)

            test_loss += loss.detach().cpu().item()

            tp += torch.eq(output.argmax(1) + impact, 2).sum().detach().cpu().item()
            cur_scores = output.detach().cpu().numpy()
            cur_predictions = output.argmax(1).detach().cpu().numpy()
            preds += sum(cur_predictions)
            predictions.extend(cur_predictions)
            scores.append(cur_scores)
            targs += impact.sum().detach().cpu().item()
            cur_targets = impact.detach().cpu().numpy()
            targets.extend(cur_targets)
            #print(impact.detach().cpu().numpy())
            n_batches += 1
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Valid prec:', tp / (preds + 0.00001), '|', 'Valid recall:', tp / (targs + 0.00001))
    logits = np.concatenate(scores, axis=0)
    competition_scores = impact_f1_metric.evaluate(unique_ids, predictions, logits)
    print('Competition metric', competition_scores)
    print('Thresh F1 metric', thresh_f1_metric.evaluate(targets, logits))

    print('Validation time in %d minutes, %d seconds' % (mins, secs))
    return test_loss / n_batches, 2 * tp / (preds + targs + 0.00001),  competition_scores[0][2]


def train_epochs(model, loaders, epochs, criterion, optimizer, scheduler=None, device='cuda:0'):
    def train_one_epoch(dataloader):
        model.train()
        train_loss = 0
        tp, preds, targs = 0, 0, 0
        n_batches = 0
        for i, item in enumerate(Bar(dataloader)):
            optimizer.zero_grad()
            data, impact = item[IMAGE_KEY].to(device), item[LABELS_KEY].to(device)
            output = model(data)
            loss = criterion(output, impact)

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss += loss.detach().cpu().item()
                tp += torch.eq(output.argmax(1) + impact, 2).sum().detach().cpu().item()
                preds += output.argmax(1).sum().detach().cpu().item()
                targs += impact.sum().detach().cpu().item()
        n_batches += 1
        if scheduler:
            scheduler.step()
        print('\t Train prec:', tp / (preds + 0.00001), '|', '\t Train recall:', tp / (targs + 0.00001))
        return train_loss / n_batches, 2 * tp / (preds + targs + 0.00001)

    # model.load_state_dict(torch.load('videoclf_best_f1_0.57.bin'))
    #model_fp = 'videoclf_{:03d}ep_{:.1f}f1_{:.6f}loss.bin'
    impact_f1_metric = ImpactF1()
    thresh_f1_metric = ThreshF1()

    model_best_loss_fp = 'videoclf_best_loss.bin'
    model_best_f1_fp = 'videoclf_best_f1.bin'

    train_dataloader = loaders['train']
    valid_dataloader = loaders['valid']

    best_loss = float('inf')
    best_f1 = -1

    #try:
    print('Start training')
    for epoch in range(epochs):
        print('\nEPOCH:', {epoch}, '\n')
        start_time = time.time()
        train_loss, train_f1 = train_one_epoch(train_dataloader)
        print(f'\tLoss: {train_loss:.7f}(train)\t|\tF1: {train_f1 * 100:.1f}%(train)')

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60
        print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))

        valid_loss, valid_f1, comp_f1 = test(valid_dataloader, model, criterion, impact_f1_metric, thresh_f1_metric, device)

        if comp_f1 > best_f1:
            torch.save(model.state_dict(), model_best_f1_fp)
            best_f1 = comp_f1

        if valid_loss < best_loss:
            torch.save(model.state_dict(), model_best_loss_fp)
            best_loss = valid_loss

        print(f'Loss: {valid_loss:.7f}(valid) | F1: {valid_f1 * 100:.1f}%(valid)')
    #except KeyboardInterrupt:
    #    raise Exception('Train loop canceled')
    #finally:
    if epochs != 1:
        print('Checking the results of test dataset...')
        model.load_state_dict(torch.load(model_best_f1_fp))
        model.eval()
        test_loss, test_f1, comp1 = test(valid_dataloader, model, criterion, impact_f1_metric, thresh_f1_metric, device)
        print(f'Loss: {test_loss:.7f}(test) | F1: {test_f1 * 100:.1f}%(test)')

