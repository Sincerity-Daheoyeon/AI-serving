import os
import torch
from transformers import SwinForImageClassification

def return_model_by_type(type):
    if type == 'Classification':
        model_name = "microsoft/swin-tiny-patch4-window7-224"
        weights_dir = os.getenv("WEIGHTS_DIR")
        weights_file = os.getenv("500_swin_model_weights.pth")
        weights_path = os.path.join(weights_dir, weights_file)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SwinForImageClassification.from_pretrained(
                model_name,
                num_labels=2,
                ignore_mismatched_sizes=True
        )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")

        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
    else:
        print("There is not Model requseted. requested type : ",type)
    return model