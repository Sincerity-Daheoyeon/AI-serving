import os
import torch
from transformers import SwinForImageClassification

def return_model_by_type(type):
    if type == 'Classification':
        model_name = "microsoft/swin-tiny-patch4-window7-224"
        weights_path = "./WEIGHTS_DIR/500_swin_model_weights.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SwinForImageClassification.from_pretrained(
                model_name,
                num_labels=2,
                ignore_mismatched_sizes=True
        )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")

        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.to(device)
        
        dummy_input = torch.rand(1, 3, 224, 224).to(device)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)  # 모델 출력
        print("Model output shape:", output.logits.shape)  # 출력의 shape 확인
    else:
        print("There is not Model requseted. requested type : ",type)
    return model