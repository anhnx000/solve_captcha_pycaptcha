from model.model import captcha_model, model_conv, model_resnet
from utils.arg_parsers import predict_arg_parser
from data.dataset import str_to_vec, lst_to_str

import torchvision.transforms as transforms
from PIL import Image



transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict(args):
    model = captcha_model.load_from_checkpoint(args.ckpt, model=model_resnet(), use_ctc=args.use_ctc)
    model.eval()
    
    img = transform(Image.open(args.input))
    img = img.unsqueeze(0)
    y = model(img)
    
    # Handle the output based on whether CTC is used
    if args.use_ctc:
        # For CTC output processing
        import torch.nn.functional as F
        log_probs = F.log_softmax(y, dim=2)
        y = log_probs.permute(1, 0, 2)
    else:
        # Original processing
        y = y.permute(1, 0, 2)
        
    pred = y.argmax(dim=2)

    ans = lst_to_str(pred)
    print(ans)
    return ans


if __name__ == "__main__":
    args = predict_arg_parser()
    predict(args)
    

