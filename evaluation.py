import os
import json
import torch
import numpy as np
from loss import jaccard_sim
from scipy.spatial import distance
from basic.generic_utils import Progbar
from basic.common import makedirsforfile

def l2norm(X):
    """
    L2-normalize columns of X
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / norm


def cal_error(videos, captions, measure='cosine'):
    if measure == 'cosine':
        captions = l2norm(captions)
        videos = l2norm(videos)
        errors = -1*np.dot(captions, videos.T)
    elif measure == 'euclidean':
        errors = distance.cdist(captions, videos, 'euclidean')
    elif measure == 'l1':
        errors = distance.cdist(captions, videos, 'minkowski', p=1)
    elif measure == 'l2':
        errors = distance.cdist(captions, videos, 'euclidean')
    elif measure == 'l1_norm':
        errors = - distance.cdist(captions, videos, 'minkowski', p=1)/videos.shape[1]-1
    elif measure == 'l2_norm':
        errors = - distance.cdist(captions, videos, 'euclidean')/videos.shape[1]-1
    elif measure == 'jaccard':
        captions = torch.Tensor(captions)
        videos = torch.Tensor(videos)
        errors = -1*jaccard_sim(captions, videos)
        errors = errors.numpy()
    return errors


# if the number of videos or captions are too large, the memory may be not enough for jaccard similarity computation.
# Hence, we split the sentence embedding matrix into a sequence of matrices with smaller size
def cal_error_batch(videos, captions, measure='cosine', batch_size=2000):
    if measure == 'cosine':
        captions = l2norm(captions)
        videos = l2norm(videos)
        errors = -1*np.dot(captions, videos.T)
    elif measure == 'euclidean':
        errors = distance.cdist(captions, videos, 'euclidean')
    elif measure == 'l1':
        errors = distance.cdist(captions, videos, 'minkowski', p=1)
    elif measure == 'l2':
        errors = distance.cdist(captions, videos, 'euclidean')
    elif measure == 'l1_norm':
        errors = - distance.cdist(captions, videos, 'minkowski', p=1)/videos.shape[1]-1
    elif measure == 'l2_norm':
        errors = - distance.cdist(captions, videos, 'euclidean')/videos.shape[1]-1
    elif measure == 'jaccard':
        idx = 0
        errors = None
        while 1:
            # print(idx)
            sub_captions = captions[idx*batch_size:(idx+1)*batch_size,:]
            sub_captions = torch.Tensor(sub_captions)
            videos = torch.Tensor(videos)
            sub_errors = -1*jaccard_sim(sub_captions, videos)
            if errors is None:
                errors = sub_errors.numpy()
            else:
                errors = np.append(errors, sub_errors, axis=0)
            if (idx+1)*batch_size > captions.shape[0]:
                break
            idx=idx+1
    return errors


def cal_simi(captions, videos, measure='cosine'):
    if measure == 'cosine':
        captions = l2norm(captions)
        videos = l2norm(videos)
        errors = np.dot(captions, videos.T)
    elif measure == 'jaccard':
        captions = torch.Tensor(captions)
        videos = torch.Tensor(videos)
        errors = jaccard_sim(captions, videos)
    return errors


# predict tags
def pred_tag(tag_prob_embs, video_ids, tag_vocab_path, output_dir, k=10):
    tag_vocab_list = json.load(open(tag_vocab_path, 'r'))
    tag_vocab_size = len(tag_vocab_list)
    idx2tag = dict(zip(range(tag_vocab_size), tag_vocab_list))
    # print(tag_prob_embs.shape)
    assert tag_prob_embs.shape[1] == tag_vocab_size, "%s != %s" % (tag_prob_embs.shape[1], tag_vocab_size)
    
    output_file = os.path.join(output_dir, 'pred_tags.txt')
    makedirsforfile(output_file)
    fout = open(output_file, 'w')

    for idx, prob_vec in enumerate(tag_prob_embs):
        vid_id = video_ids[idx]
        top_hits = np.argsort(prob_vec)[::-1][:k]
        fout.write(vid_id + '\t')
        for hit in top_hits:
            fout.write("%s:%.3f " % (idx2tag[hit], prob_vec[hit]))
        fout.write('\n')
    fout.close()


# encode text or video
def encode_text_or_vid(encoder, data_loader, return_ids=True):
    """
    Encode all videos and captions loadable by `data_loader`
    """
    # numpy array to keep all the embeddings
    embeddings = None
    ids = ['']*len(data_loader.dataset)
    pbar = Progbar(len(data_loader.dataset))
    for i, (datas, idxs, data_ids) in enumerate(data_loader):

        # compute the embeddings
        emb = encoder(datas)

        # initialize the numpy arrays given the size of the embeddings
        if embeddings is None:
            embeddings = np.zeros((len(data_loader.dataset), emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        embeddings[idxs] = emb.data.cpu().numpy().copy()

        for j, idx in enumerate(idxs):
            ids[idx] = data_ids[j]

        del datas
        pbar.add(len(idxs))

    if return_ids == True:
        return embeddings, ids,
    else:
        return embeddings


# encode text or video in hybrid manner
def encode_text_or_vid_tag_hist_prob(encoder, data_loader, return_ids=True):
    """
    Encode all videos and captions loadable by `data_loader`
    """
    # numpy array to keep all the embeddings
    init_flag = True
    ids = ['']*len(data_loader.dataset)
    pbar = Progbar(len(data_loader.dataset))
    for i, (datas, idxs, data_ids) in enumerate(data_loader):

        # compute the embeddings
        (global_emb, local_emb, temporal_emb, temp_spa_emb), (global_tag_prob, local_tag_prob, temporal_tag_prob, temp_spa_tag_prob) = encoder(datas)

        # initialize the numpy arrays given the size of the embeddings
        if init_flag:
            init_flag = False
            if global_emb is not None:
                global_embeddings = np.zeros((len(data_loader.dataset), global_emb.size(1)))
                local_embeddings = np.zeros((len(data_loader.dataset), local_emb.size(1)))
                temporal_embeddings = np.zeros((len(data_loader.dataset), temporal_emb.size(1)))
                temp_spa_embeddings = np.zeros((len(data_loader.dataset), temp_spa_emb.size(1)))
            else:
                global_embeddings = None
            if global_tag_prob is not None:
                global_tag_prob_embs = np.zeros((len(data_loader.dataset), global_tag_prob.size(1)))
                local_tag_prob_embs = np.zeros((len(data_loader.dataset), local_tag_prob.size(1)))
                temporal_tag_prob_embs = np.zeros((len(data_loader.dataset), temporal_tag_prob.size(1)))
                temp_spa_tag_prob_embs = np.zeros((len(data_loader.dataset), temp_spa_tag_prob.size(1)))
            else:
                global_tag_prob_embs = None

        # preserve the embeddings by copying from gpu and converting to numpy
        if global_tag_prob_embs is not None:
            global_embeddings[idxs] = global_emb.data.cpu().numpy().copy()
            local_embeddings[idxs] = local_emb.data.cpu().numpy().copy()
            temporal_embeddings[idxs] = temporal_emb.data.cpu().numpy().copy()
            temp_spa_embeddings[idxs] = temp_spa_emb.data.cpu().numpy().copy()
        if global_tag_prob is not None:
            global_tag_prob_embs[idxs] = global_tag_prob.data.cpu().numpy().copy()
            local_tag_prob_embs[idxs] = local_tag_prob.data.cpu().numpy().copy()
            temporal_tag_prob_embs[idxs] = temporal_tag_prob.data.cpu().numpy().copy()
            temp_spa_tag_prob_embs[idxs] = temp_spa_tag_prob.data.cpu().numpy().copy()

        for j, idx in enumerate(idxs):
            ids[idx] = data_ids[j]

        del datas
        pbar.add(len(idxs))

    if return_ids == True:
        return (global_embeddings, local_embeddings, temporal_embeddings, temp_spa_embeddings), (global_tag_prob_embs, local_tag_prob_embs, temporal_tag_prob_embs, temp_spa_tag_prob_embs), ids,
    else:
        return (global_embeddings, local_embeddings, temporal_embeddings, temp_spa_embeddings), (global_tag_prob_embs, local_tag_prob_embs, temporal_tag_prob_embs, temp_spa_tag_prob_embs)