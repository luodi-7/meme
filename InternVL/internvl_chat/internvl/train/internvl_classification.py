from transformers.modeling_utils import PreTrainedModel
from internvl.model.internvl_chat import (InternVLChatConfig,
                                          InternVLChatModel)
import torch
import torch.nn as nn

# class InternVLForSequenceClassification(PreTrainedModel):
#     def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, num_labels=None):
#         super().__init__(config)
#         self.internvlchat = InternVLChatModel(config, vision_model, language_model)
#         self.classifier = nn.Linear(self.internvlchat.hidden_size, num_labels)
    
#     def forward(
#             self,
#             pixel_values: torch.FloatTensor,
#             input_ids: torch.LongTensor = None,
#             attention_mask = None,
#             position_ids = None,
#             image_flags = None,
#             past_key_values = None,
#             labels = None,
#             use_cache = None,
#             output_attentions = None,
#             output_hidden_states = None,
#             return_dict = None,
#     ):
#         outputs = self.internvlchat(
#             pixel_values = pixel_values,
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             image_flags=image_flags,
#             past_key_values=past_key_values,
#             labels=labels,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict)
#         pooled_output = outputs[1]
#         return self.classifier(pooled_output)


class InternVLForSequenceClassification(InternVLChatModel):
    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, num_labels_list=[None]):
        super().__init__(config, vision_model, language_model)
        self.num_tasks = len(num_labels_list)
        self.classifier_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, num_labels) for num_labels in num_labels_list
        ])
    #     self._convert_to_bfloat16()

    # def _convert_to_bfloat16(self):
    #     for layer in self.classifier_heads:
    #         # layer.weight.data = layer.weight.data.to(torch.bfloat16)
    #         # if layer.bias is not None:
    #         #     layer.bias.data = layer.bias.data.to(torch.bfloat16)
    #         #layer.weight = layer.weight.to(torch.bfloat16)
    #         # nn.init.uniform_(layer.weight.data)
    #         # nn.init.zeros_(layer.bias.data)

    #         print(f'weight: {layer.weight}, shape: {layer.weight.shape}')
    #         print(f'bias: {layer.bias}')

    #     print('//////////////////////')
    #     for layer in self.classifier_heads:


    #         print(f'weight: {layer.weight}, shape: {layer.weight.shape}')
    #         print(f'bias: {layer.bias}')

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask = None,
            position_ids = None,
            image_flags = None,
            past_key_values = None,
            labels = None,
            label_class = None,
            use_cache = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
    ):
        outputs = super().forward(
            pixel_values = pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        pooled_output = outputs.hidden_states[-1] 

        pooled_output=pooled_output.to(torch.bfloat16)
        
        logits = []
        for classifier_head in self.classifier_heads:
            task_logits = classifier_head(pooled_output)
            logits.append(task_logits)

            # print(f'Task logits: {task_logits}')
        
        return logits