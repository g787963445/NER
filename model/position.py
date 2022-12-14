# pinyin_bert
import torch


class My_BertModel(BertModel):

    def __init__(self, config):
        super(My_BertModel, self).__init__(config)
        self.config = config

        self.embeddings = All_embeding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            pinyin_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long, device=device)
        if position_ids is None:
            max = 0
            position_ids1 = []
            position_ids = []
            for i in attention_mask.to(torch.device("cpu")).numpy().tolist():
                if max < len(i):
                    max = len(i)
                position_ids2 = []
                t = 1
                for i1 in i:
                    if i1 == 1:
                        position_ids2.append(t)
                        t = t + 1
                    else:
                        break
                position_ids1.append(position_ids2)
            for i in position_ids1:
                position_ids.append(i + [0] * (max - len(i)))
            position_ids = torch.tensor(position_ids, device=device)
        bs, sentence_len = attention_mask.size()
        extend_attention_mask = attention_mask.view(bs, 1, 1, sentence_len)
        embedding_output = self.embeddings(
            input_ids=input_ids, pinyin_ids=pinyin_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extend_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

