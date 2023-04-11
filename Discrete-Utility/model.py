
from transformers import BertForMaskedLM
import torch.nn as nn


class BERTPrompt4NR(nn.Module):
    def __init__(self, model_name, answer_ids, args):
        super(BERTPrompt4NR, self).__init__()
        self.BERT = BertForMaskedLM.from_pretrained(model_name)
        self.BERT.resize_token_embeddings(args.vocab_size)

        for param in self.BERT.parameters():
            param.requires_grad = True

        self.answer_ids = answer_ids
        self.mask_token_id = 103
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, batch_enc, batch_attn, batch_labs):
        outputs = self.BERT(input_ids=batch_enc,
                            attention_mask=batch_attn)
        out_logits = outputs.logits

        mask_position = batch_enc.eq(self.mask_token_id)
        mask_logits = out_logits[mask_position, :].view(out_logits.size(0), -1, out_logits.size(-1))[:, -1, :]

        answer_logits = mask_logits[:, self.answer_ids]

        loss = self.loss_func(answer_logits, batch_labs)

        return loss, answer_logits.softmax(dim=1)
