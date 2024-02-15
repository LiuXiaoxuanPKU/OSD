import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F



import transformers
from transformers import Trainer, BitsAndBytesConfig, PretrainedConfig
from transformers.trainer_pt_utils import LabelSmoother
from transformers import LlamaModel, LlamaForCausalLM

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download

from operator import itemgetter

torch.set_printoptions(profile="full")

def predictor_forward(input_ids, model, past_key_values):
    """
    This function performs the following operations:
    1. Forward pass through the model to obtain the predictor logits, original model outputs, and logits.

    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model (PredictorLMHead): The model containing the MLP predictor layers and base model.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - predictor_values (torch.Tensor): predictor values from the Predictor heads.
    - logits (torch.Tensor): Original logits from the base model.
    """
    predictor_values, outputs, logits = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )

    return predictor_values, logits


class PredictorOSDConfig(PretrainedConfig):
    """
    Configuration class for OSD model.

    Args:
        predictor_num_heads (int, optional): Number of heads for the Predictor layer. Default is 1.
        predictor_num_layers (int, optional): Number of Predictor layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.5".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        predictor_num_heads=1,
        predictor_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.5",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.predictor_num_heads = predictor_num_heads
        self.predictor_num_layers = predictor_num_layers
        self.base_model_name_or_path = base_model_name_or_path


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class PredictorModel(nn.Module):
    """The OSD Predictor Language Model Head.

    This module creates a series of prediction heads (based on the 'predictor' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
        self,
        base_model,
        teacher_model=None,
        predictor_num_heads=1,
        predictor_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.5",
    ):
        """
        Args:
            base_model (nn.Module): The base language model to be used.
            predictor_num_heads (int, optional): Number of tokens to predict. Defaults to 1.
            predictor_num_layers (int, optional): Number of ResBlock layers for each Predictor head. Defaults to 1.
        """
        super().__init__()
        self.base_model = base_model
        self.teacher_model = teacher_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]  
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.predictor = predictor_num_heads
        self.predictor_num_layers = predictor_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        # Create a list of Predictor heads
        self.predictor_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * predictor_num_layers),
                    nn.Linear(self.hidden_size, 1, bias=True),
                    nn.Sigmoid()
                )
                for _ in range(predictor_num_heads)
            ]
        )

        # Ensure predictor_head's dtype and device align with the base_model
        self.predictor_head.to(self.base_model.dtype).to(self.base_model.device)

        for i in range(predictor_num_heads):
            # Random initialization of the linear layers
            torch.nn.init.uniform_(self.predictor_head[i][:-1][0].linear.weight.data[:])
            torch.nn.init.uniform_(self.predictor_head[i][:-1][1].weight.data[:])

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        predictor_head_name_or_path,
        base_model=None,
        teacher_model=None,
        predictor_num_heads=None,
        **kwargs,
    ):
        """
        Args:
            predictor_head_name_or_path (str): Name or path of the Predictor head to load.
            **kwargs: Additional keyword arguments for loading the base model.

        Returns:
            PredictorModel: A PredictoraModel instance loaded from the given path.
        """
        predictor_config = PredictorOSDConfig.from_pretrained(predictor_head_name_or_path)
        if predictor_num_heads is not None:
            print("Overriding predictor_num_heads as:", predictor_num_heads)
            predictor_config.predictor_num_heads = predictor_num_heads
        if base_model is not None:
            print("Overriding base_model as:", base_model)
            predictor_config.base_model_name_or_path = base_model
            
        base_model = LlamaForCausalLM.from_pretrained(
            predictor_config.base_model_name_or_path, **kwargs
        )

        model = cls(
            base_model,
            teacher_model,
            predictor_config.predictor_num_heads,
            predictor_config.predictor_num_layers,
            predictor_config.base_model_name_or_path,
        )
        predictor_head_path = os.path.join(predictor_head_name_or_path, "predictor_lm_head.pt")
        if os.path.exists(predictor_head_path):
            filename = predictor_head_path
        else:
            filename = hf_hub_download(predictor_head_name_or_path, "predictor_lm_head.pt")
        predictor_head_state_dict = torch.load(filename, map_location=base_model.device)
        model.predictor_head.load_state_dict(predictor_head_state_dict, strict=False)

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
    ):
        """Forward pass of the Predictor OSD Model.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Predictor heads.
            (Optional) Original predictions from the base model's LM head.
        """
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            orig = self.base_model.lm_head(outputs[0])
        # Clone the output hidden states
        hidden_states = outputs[0].clone()
        prompt_lens = [torch.sum(attention_mask[i]).item() for i in range(input_ids.shape[0])]
        
        new_tokens = []
        for i in range(orig.shape[0]):
            new_token = torch.argmax(torch.softmax(orig[i, -1, :] / 0.001, dim=-1), dim=-1)
            new_tokens.append(new_token)
       
        # create new input_ids, also as the outputs
        critical_dim, critical_len = max(enumerate(prompt_lens), key=itemgetter(1))
        # expand input_ids by a length of 1 along dim -1
        zeros = torch.zeros(input_ids.shape[0], 1).to(input_ids.device, dtype=int)
        input_ids = torch.cat((input_ids, zeros), dim=1)

        # add new token to left-padded sequence
        for i in range(input_ids.shape[0]):
            input_ids[i, -1] = new_tokens[i]
        
        # extract last token
        extracted_states = None
        for p in range(len(prompt_lens)):
            if p == 0:
                extracted_states = hidden_states[p, -1, :].unsqueeze(0)
            else:
                # cat along the batch size dim
                extracted_states = torch.cat((extracted_states, hidden_states[p, -1, :].unsqueeze(0)), dim=0)
        
        #print(extracted_states.shape)
        predictor_values = []
        # TODO: Consider parallelizing this loop for efficiency
        for i in range(self.predictor):
            predictor_values.append(self.predictor_head[i](extracted_states))
        
        if output_orig:
            return torch.stack(predictor_values, dim=0), input_ids, outputs, orig
        return torch.stack(predictor_values, dim=0), input_ids