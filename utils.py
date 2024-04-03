from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
def rouge(reference, candidate, types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True,
          split_summaries=True):
    """
    This is based on rouge-score 0.0.4
    If using rougeLsum, it is necessary to split sentences with '\n' in summaries in advance
    """
    if 'rougeLsum' in types and split_summaries:
        reference = '\n'.join(sent_tokenize(reference))
        candidate = '\n'.join(sent_tokenize(candidate))

    results = {}
    for t in types:
        if t not in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
            print("The type must be selected in rouge1, rouge2, rougeL, and rougeLsum.")
            return results
    scorer = rouge_scorer.RougeScorer(types, use_stemmer=use_stemmer)
    scores = scorer.score(reference, candidate)
    for t in types:
        r = {}
        r["precision"] = scores[t].precision
        r["recall"] = scores[t].recall
        r["fmeasure"] = scores[t].fmeasure
        results[t] = r
    return results


def rouge_corpus(references, candidates, types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True,
                 split_summaries=True):
    if len(references) != len(candidates):
        print("len must be equal")
        return None
    results = {}
    for t in types:
        s = {}
        s['recall'] = []
        s['precision'] = []
        s['fmeasure'] = []
        results[t] = s
    for ref, can in zip(references, candidates):
        s = rouge(ref, can, types=types, use_stemmer=use_stemmer, split_summaries=split_summaries)
        for t in types:
            results[t]['recall'].append(s[t]['recall'])
            results[t]['precision'].append(s[t]['precision'])
            results[t]['fmeasure'].append(s[t]['fmeasure'])

    final_results = {}
    for t in types:
        s = results[t]
        tmp = {}
        tmp['precision'] = np.mean(s['precision'])
        tmp['recall'] = np.mean(s['recall'])
        tmp['fmeasure'] = np.mean(s['fmeasure'])
        final_results[t] = tmp
    return final_results