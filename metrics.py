import os
import config
import logging

def get_entity_bios(seq):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [('PER', 0,1), ('LOC', 3, 3)]
    """
    chunks = []
    chunk = [-1, -1, -1]

    for indx, tag in enumerate(seq):
        # if not isinstance(tag, str):
        #     tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entity_bio(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]

    for indx, tag in enumerate(seq):
        # if not isinstance(tag, str):
        #     tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entities(seq,markup='bios'):
    '''
    :param seq:
    :param markup:
    :return:
    '''
    assert markup in ['bio','bios']
    if markup =='bio':
        return get_entity_bio(seq)
    else:
        return get_entity_bios(seq)


def f1_score(y_true, y_pred, mode='dev'):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        f1_score(y_true, y_pred)
        0.50
    """
    true_entities=[]
    pred_entities=[]
    for indx,_ in enumerate(y_true):
        true_entities=true_entities+get_entities(y_true[indx],'bio')
        pred_entities=pred_entities+get_entities(y_pred[indx],'bio')

    true_entities = set([tuple(t) for t in true_entities])
    pred_entities = set([tuple(t) for t in pred_entities])
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p_all = nb_correct / nb_pred if nb_pred > 0 else 0
    r_all = nb_correct / nb_true if nb_true > 0 else 0
    score_all = 2 * p_all * r_all / (p_all + r_all) if p_all + r_all > 0 else 0
    if mode == 'dev':
        return p_all, r_all, score_all # change score to [p,r,score]
    else:
        f_score = {}
        for label in config.labels:
            true_entities_label = set()
            pred_entities_label = set()
            for t in true_entities:
                if t[0] == label:
                    true_entities_label.add(t)
            for p in pred_entities:
                if p[0] == label:
                    pred_entities_label.add(p)
            nb_correct_label = len(true_entities_label & pred_entities_label)
            nb_pred_label = len(pred_entities_label)
            nb_true_label = len(true_entities_label)

            p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
            r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
            score_label = 2 * p_label * r_label / (p_label + r_label) if p_label + r_label > 0 else 0
            f_score[label] = [p_label,r_label,score_label] #change score_label to [p_label,r_label,score_label]
        return f_score, p_all, r_all, score_all  # chagne score to p,r,score


def bad_case(y_true, y_pred, data):
    if not os.path.exists(config.case_dir):
        os.system(r"touch {}".format(config.case_dir))  # invoke the system command line to create a file
    with open(config.case_dir, 'w',encoding='utf-8') as output:
        for idx, (t, p) in enumerate(zip(y_true, y_pred)):
            if t == p:
                continue
            else:
                output.write("bad case " + str(idx) + ": \n")
                output.write("sentence: " + str(data[idx]) + "\n")
                output.write("golden label: " + str(t) + "\n")
                output.write("model pred: " + str(p) + "\n")
    logging.info("--------Bad Cases reserved !--------")


if __name__ == "__main__":
    y_t = [['O', 'O', 'O', 'B-address', 'I-address', 'I-address', 'O'], ['B-name', 'I-name', 'O']]
    y_p = [['O', 'O', 'B-address', 'I-address', 'I-address', 'I-address', 'O'], ['B-name', 'I-name', 'O']]
    sent = [['十', '一', '月', '中', '山', '路', '电'], ['周', '静', '说']]
    bad_case(y_t, y_p, sent)
