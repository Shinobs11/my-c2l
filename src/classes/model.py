import torch, typing
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional


class BertForCounterfactualRobustness(BertForSequenceClassification):
  def __init__(self, config):
    super().__init__(config)
    

  def forward(
    self,
    anchor_input_ids: Optional[torch.Tensor] = None,
    positive_input_ids: Optional[torch.Tensor] = None,
    negative_input_ids: Optional[torch.Tensor] = None,
    anchor_attention_mask:Optional[torch.Tensor] = None,
    positive_attention_mask: Optional[torch.Tensor] = None,
    negative_attention_mask: Optional[torch.Tensor] = None,
    anchor_token_type_ids: Optional[torch.Tensor] = None,
    positive_token_type_ids: Optional[torch.Tensor] = None,
    negative_token_type_ids: Optional[torch.Tensor] = None,
    triplet_sample_masks: Optional[torch.Tensor] = None,
    lambda_weight: Optional[float] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
  ):
    
    anchor_outputs = self.bert(
      input_ids=anchor_input_ids,
      attention_mask=anchor_attention_mask,
      token_type_ids=anchor_token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    pooled_output = anchor_outputs[1]
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    loss = None




    # loss_fct = torch.nn.BCEWithLogitsLoss()
    # loss = loss_fct(logits, labels)



    if labels is not None:
      if self.config.problem_type is None:
          if self.num_labels == 1:
              self.config.problem_type = "regression"
          elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
              self.config.problem_type = "single_label_classification"
          else:
              self.config.problem_type = "multi_label_classification"

      if self.config.problem_type == "regression":
          loss_fct = torch.nn.MSELoss()
          if self.num_labels == 1:
              loss = loss_fct(logits.squeeze(), labels.squeeze())
          else:
              loss = loss_fct(logits, labels)
      elif self.config.problem_type == "single_label_classification":
          loss_fct = torch.nn.CrossEntropyLoss()
          
          if labels.shape[-1] != 1 and len(labels.shape) > 1:
            labels = labels.argmax(dim=-1)
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      elif self.config.problem_type == "multi_label_classification":
          loss_fct = torch.nn.BCEWithLogitsLoss()
          loss = loss_fct(logits, labels)
          
    if positive_input_ids is not None and negative_input_ids is not None and labels is not None:
      # positive_outputs = self.bert(
      #   positive_input_ids,
      #   attention_mask=positive_attention_mask,
      #   token_type_ids=positive_token_type_ids,
      #   position_ids=position_ids,
      #   head_mask=head_mask,
      #   inputs_embeds=inputs_embeds,
      #   output_attentions=output_attentions,
      #   output_hidden_states=output_hidden_states,
      #   return_dict=return_dict,
      # )
      # negative_outputs = self.bert(
      #     negative_input_ids,
      #     attention_mask=negative_attention_mask,
      #     token_type_ids=negative_token_type_ids,
      #     position_ids=position_ids,
      #     head_mask=head_mask,
      #     inputs_embeds=inputs_embeds,
      #     output_attentions=output_attentions,
      #     output_hidden_states=output_hidden_states,
      #     return_dict=return_dict,
      # )
      
      triplet_loss = None
      triplet_loss_func = torch.nn.TripletMarginLoss()

      if lambda_weight is None:
        lambda_weight = 0.1
      
      if triplet_sample_masks is None:
        # triplet_loss = triplet_loss_func(anchor_outputs[1], positive_outputs[1], negative_outputs[1])
        triplet_loss = triplet_loss_func(anchor_outputs[1], anchor_outputs[1], anchor_outputs[1])
        loss = loss + 0.0 * triplet_loss
      else:
        if torch.sum(triplet_sample_masks):
          # triplet_loss = triplet_loss_func(anchor_outputs[1][triplet_sample_masks], positive_outputs[1][triplet_sample_masks], negative_outputs[1][triplet_sample_masks])
          triplet_loss = triplet_loss_func(anchor_outputs[1][triplet_sample_masks], anchor_outputs[1][triplet_sample_masks], anchor_outputs[1][triplet_sample_masks])
          loss = loss + 0.0 * triplet_loss
    


    if not return_dict:
        output = (logits,) + anchor_outputs[2:]
        return ((loss,) + output) if loss is not None else output





    return SequenceClassifierOutput(
        loss=loss,
        logits=logits
    )