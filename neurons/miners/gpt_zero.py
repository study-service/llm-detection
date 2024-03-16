"""
This code a slight modification of perplexity by hugging face
https://huggingface.co/docs/transformers/perplexity

Both this code and the orignal code are published under the MIT license.

by Burhan Ul tayyab and Nicholas Chua
"""

import torch
import re
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForSequenceClassification
from collections import OrderedDict
import bittensor as bt


class GPT2PPL:
    # model_id
    # Hello-SimpleAI/HC3
    # gpt2
    # gpt2-medium
    def __init__(self, device="cuda", model_id="allenai/c4"):
        bt.logging.info(f"Running model_id = {model_id}")
        self.device = device
        self.model_id = model_id
        kwargs = {
            "use_auth_token": "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"
        }
        self.model = GPT2LMHeadModel.from_pretrained(model_id, **kwargs).to(device)
        # self.model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/HC3")

        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id, **kwargs)

        self.max_length = self.model.config.n_positions
        self.stride = 512

    def __call__(self, sentence):
        ppl = self.getPPL(sentence)
        if ppl is None:
            bt.logging.info('Got None PPL on text "{}..."'.format(sentence))
            return 0

        return (100 - ppl) / 100

    def getPPL(self, sentence):
        encodings = self.tokenizer(sentence, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        likelihoods = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
                likelihoods.append(neg_log_likelihood)

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        if torch.isnan(torch.Tensor(nlls)).any() or len(nlls) == 0:
            return None

        ppl = int(torch.exp(torch.stack(nlls).sum() / end_loc))
        return ppl

        # used fix bug by gpt
        # encodings = self.tokenizer(sentence, return_tensors="pt")
        # input_ids = encodings.input_ids.to(self.device)
        # with torch.no_grad():
        #     outputs = self.model(input_ids)
        #     logits = outputs.logits
        #     # Calculate perplexity
        #     log_probs = torch.log_softmax(logits, dim=-1)
        #     token_likelihoods = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        #     perplexity = torch.exp(-token_likelihoods.mean())
        # return perplexity.item()


if __name__ == '__main__':
    model = GPT2PPL(device='cpu')
    text = 'Hello world, i am here'
    res = model(text)
    print(res)
