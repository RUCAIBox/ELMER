import math
import torch
import json
import multiprocessing
from multiprocessing import Pool, cpu_count
from transformers import BartConfig, BartTokenizer, BartForConditionalGeneration
from fairseq.data import data_utils


BOS_ID, PAD_ID, EOS_ID, MASK_ID = 0, 1, 2, 50264
full_stop_index = 4
random_ratio = 0.1
replace_length = 1

mask_span_distribution = None
_lambda = 3.5
lambda_to_the_k = 1
e_to_the_minus_lambda = math.exp(-_lambda)
k_factorial = 1
ps = []
for k in range(0, 128):
    ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
    lambda_to_the_k *= _lambda
    k_factorial *= k + 1
    if ps[-1] < 0.0000001:
        break
ps = torch.FloatTensor(ps)
mask_span_distribution = torch.distributions.Categorical(ps)

tokenizer = BartTokenizer.from_pretrained("pretrained_model/bart-base")


def gather_word_starts(source, mask_whole_word):
    if mask_whole_word is not None:
        is_word_start = mask_whole_word.gather(0, source)
    else:
        is_word_start = torch.ones(source.size())
    is_word_start[0] = 0
    is_word_start[-1] = 0
    return is_word_start


def get_whole_word_mask(dictionary):

    def is_beginning_of_word(i):
        tok = dictionary.decoder[i]
        if tok.startswith("Ġ"):
            return True
        else:
            return False

    mask_whole_words = torch.ByteTensor(
        list(map(is_beginning_of_word, range(len(tokenizer))))
    )
    return mask_whole_words


def add_insertion_noise(tokens, p):
    if p == 0.0:
        return tokens

    num_tokens = len(tokens)
    n = int(math.ceil(num_tokens * p))

    noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
    noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
    noise_mask[noise_indices] = 1
    result = torch.LongTensor(n + len(tokens)).fill_(-1)

    num_random = int(math.ceil(n * random_ratio))
    result[noise_indices[num_random:]] = MASK_ID
    result[noise_indices[:num_random]] = torch.randint(
        low=3, high=len(tokenizer), size=(num_random,)
    )

    result[~noise_mask] = tokens

    assert (result >= 0).all()

    return result


def add_whole_word_mask(source, p):
    mask_whole_word = get_whole_word_mask(tokenizer)
    is_word_start = gather_word_starts(source, mask_whole_word)
    num_to_mask = int(math.ceil(is_word_start.float().sum() * p))

    num_inserts = 0
    if num_to_mask == 0:
        return source

    if mask_span_distribution is not None:
        lengths = mask_span_distribution.sample(sample_shape=(num_to_mask,))
        # Make sure we have enough to mask
        cum_length = torch.cumsum(lengths, 0)
        while cum_length[-1] < num_to_mask:
            lengths = torch.cat(
                [
                    lengths,
                    mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                ],
                dim=0,
            )
            cum_length = torch.cumsum(lengths, 0)

        # Trim to masking budget
        i = 0
        while cum_length[i] < num_to_mask:
            i += 1
        lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
        num_to_mask = i + 1
        lengths = lengths[:num_to_mask]

        # Handle 0-length mask (inserts) separately
        lengths = lengths[lengths > 0]
        num_inserts = num_to_mask - lengths.size(0)
        num_to_mask -= num_inserts
        if num_to_mask == 0:
            return add_insertion_noise(source, num_inserts / source.size(0))

        assert (lengths > 0).all()
    else:
        lengths = torch.ones((num_to_mask,)).long()

    assert is_word_start[-1] == 0
    word_starts = is_word_start.nonzero(as_tuple=False)
    indices = word_starts[
        torch.randperm(word_starts.size(0))[:num_to_mask]
    ].squeeze(1)
    mask_random = torch.FloatTensor(num_to_mask).uniform_() < random_ratio

    source_length = source.size(0)
    assert source_length - 1 not in indices
    to_keep = torch.ones(source_length, dtype=torch.bool)
    is_word_start[0] = 255
    is_word_start[-1] = 255  # acts as a long length, so spans don't go over the end of doc
    if replace_length == 0:
        to_keep[indices] = 0
    else:
        # keep index, but replace it with [MASK]
        source[indices] = MASK_ID
        source[indices[mask_random]] = torch.randint(
            3, len(tokenizer), size=(mask_random.sum(),)
        )

    if mask_span_distribution is not None:
        assert len(lengths.size()) == 1
        assert lengths.size() == indices.size()
        lengths -= 1
        while indices.size(0) > 0:
            assert lengths.size() == indices.size()
            lengths -= is_word_start[indices + 1].long()
            uncompleted = lengths >= 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            lengths = lengths[uncompleted]
            if replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                source[indices] = MASK_ID
                source[indices[mask_random]] = torch.randint(
                    3, len(tokenizer), size=(mask_random.sum(),)
                )
    else:
        # A bit faster when all lengths are 1
        while indices.size(0) > 0:
            uncompleted = is_word_start[indices + 1] == 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            if replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                source[indices] = MASK_ID
                source[indices[mask_random]] = torch.randint(
                    3, len(tokenizer), size=(mask_random.sum(),)
                )

            assert source_length - 1 not in indices

    source = source[to_keep]
    if num_inserts > 0:
        source = add_insertion_noise(source, num_inserts / source.size(0))

    return source


def permute_sentences(source, p=1.0):
    full_stops = source == full_stop_index
    # Pretend it ends with a full stop so last span is a sentence
    full_stops[-2] = 1

    # Tokens that are full stops, where the previous token is not
    sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero(as_tuple=False) + 2
    document = source.clone()
    num_sentences = sentence_ends.size(0)
    num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
    substitutions = torch.randperm(num_sentences)[:num_to_permute]
    ordering = torch.arange(0, num_sentences)
    ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

    # Ignore <bos> at start
    index = 1
    for i in ordering:
        sentence = source[(sentence_ends[i - 1] if i > 0 else 1): sentence_ends[i]]
        document[index: index + sentence.size(0)] = sentence
        index += sentence.size(0)
    return document


def generate(line_list, epochs):
    part_list = [[] for _ in range(epochs)]
    for data in line_list:
        text, idx = data["text"], data["idx"]
        token_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512, return_tensors="pt")
        ids = token_ids[0]
        for epoch_idx in range(epochs):
            with data_utils.numpy_seed(123, epoch_idx, idx):
                assert ids[-1] == EOS_ID
                source, target = ids, ids.clone()
                source = permute_sentences(source, 1.0)
                source = add_whole_word_mask(source, 0.3)

            assert (source >= 0).all()
            assert (source[1:-1] >= 3).all()
            assert source[0] == BOS_ID
            assert source[-1] == EOS_ID
            part_list[epoch_idx].append({"idx": idx, "source": source.numpy().tolist(), "target": target.numpy().tolist()})
    print("complete!")
    return part_list


def split_txt(tasks, core):
    """
    切分任务
    """
    n_task = len(tasks)
    each_part = math.ceil(n_task / core)
    parts = []
    for pid in range(core):
        part = tasks[pid * each_part:(pid + 1) * each_part]
        parts.append(part)

    return parts


if __name__ == '__main__':

    data_list = []
    fin = open("data/books_wiki/books_wiki.txt", "r")
    idx = 1
    for line in fin:
        data_list.append({"text": line.strip(), "idx": idx})
        idx += 1
    fin.close()
    print("Read over bookcorpus and wikipedia.")

    core = cpu_count()
    pool = Pool()

    parts = split_txt(data_list, core)
    
    epochs = 10

    results = []
    for part in parts:
        result = pool.apply_async(generate, (part, epochs,))
        results.append(result)

    pool.close()
    pool.join()

    final_list = [[] for _ in range(epochs)]
    for result in results:
        signal = result.get()
        for i in range(epochs):
            final_list[i].extend(signal[i])

    for epoch_idx in range(epochs):
        output_file = os.path.join("data", "books_wiki", str(epoch_idx)+".json")
        fout = open(output_file, "w")
        for data in final_list[epoch_idx]:
            fout.write(json.dumps(data) + "\n")
        fout.close()

