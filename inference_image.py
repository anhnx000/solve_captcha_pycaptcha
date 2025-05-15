import os
import time
import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import argparse
from model.model import captcha_model, model_resnet, model_efficientnet, model_vit, model_mobilenet
import pytorch_lightning as pl
from data.dataset import HEIGHT, WIDTH, CLASS_NUM, CHAR_LEN, list_to_str


def parse_args():
    parser = argparse.ArgumentParser(description='Inference speed comparison between ONNX and PyTorch')
    parser.add_argument('--image_path', type=str, default=None, help='Path to the image file')
    parser.add_argument('--onnx_model_path', type=str, default=None, help='Path to the ONNX model file')
    parser.add_argument('--pytorch_model_path', type=str, default=None, help='Path to the PyTorch Lightning checkpoint')
    parser.add_argument('--model_name', type=str, default='resnet', choices=['resnet', 'efficientnet', 'vit', 'mobilenet'], 
                        help='Model architecture')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of inference runs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run inference on')
    parser.add_argument('--force_onnx_cpu', action='store_true', help='Force ONNX to run on CPU even if GPU is available')
    return parser.parse_args()

def preprocess_image(image_path):
    # This should match the preprocessing used during training
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def load_pytorch_model(model_path, model_name, device):
    # Initialize the appropriate model architecture
    if model_name == 'resnet':
        base_model = model_resnet()
    elif model_name == 'efficientnet':
        base_model = model_efficientnet()
    elif model_name == 'vit':
        base_model = model_vit()
    elif model_name == 'mobilenet':
        base_model = model_mobilenet()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Create Lightning model wrapper and load checkpoint
    model = captcha_model(model=base_model, lr=0.001)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return model

def run_pytorch_inference(model, img_tensor, device):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        # Assuming the model's forward method handles the prediction logic
        output = model(img_tensor)
    return output

def load_onnx_model(model_path, force_cpu=False):
    """
    Load ONNX model with GPU acceleration if available and not forced to use CPU
    """
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Check available providers
    available_providers = ort.get_available_providers()
    # print(f"Available ONNX Runtime providers: {available_providers}")
    
    # Choose providers based on availability and user preference
    providers = []
    
    # Add GPU providers if available and not forced to CPU
    if not force_cpu:
        # CUDA (NVIDIA GPU)
        if 'CUDAExecutionProvider' in available_providers:
            # Explicitly configure CUDA provider with optimized settings
            cuda_provider_options = {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }
            providers.append(('CUDAExecutionProvider', cuda_provider_options))
            # print("Using CUDA for ONNX inference with optimized settings")
        
        # DirectML (Windows GPU)
        elif 'DmlExecutionProvider' in available_providers:
            providers.append('DmlExecutionProvider')
            # print("Using DirectML for ONNX inference")
        
        # ROCm (AMD GPU)
        elif 'ROCMExecutionProvider' in available_providers:
            providers.append('ROCMExecutionProvider')
            # print("Using ROCm for ONNX inference")
    
    # Always add CPU as fallback
    if 'CPUExecutionProvider' in available_providers:
        providers.append('CPUExecutionProvider')
    
    # If no GPU provider was added or forced to CPU, print message
    if not providers or (providers and providers[0] == 'CPUExecutionProvider' 
                        or (isinstance(providers[0], tuple) and providers[0][0] == 'CPUExecutionProvider')):
        print("Using CPU for ONNX inference")
    
    # Create session with selected providers
    if providers:
        try:
            session = ort.InferenceSession(model_path, session_options, providers=providers)
            # Verify the provider being used
            print(f"ONNX is using: {session.get_providers()[0]}")
        except Exception as e:
            print(f"Error creating session with selected providers: {e}")
            # Fallback to CPU if there's an error
            if 'CPUExecutionProvider' in available_providers:
                print("Falling back to CPU execution")
                session = ort.InferenceSession(model_path, session_options, providers=['CPUExecutionProvider'])
            else:
                # If even CPU provider fails, try with default providers
                print("Falling back to default providers")
                session = ort.InferenceSession(model_path, session_options)
    else:
        # Fallback to default providers if none of the above are available
        session = ort.InferenceSession(model_path, session_options)
        print(f"Using default providers for ONNX inference: {session.get_providers()}")
    
    return session

def run_onnx_inference(session, img_tensor):
    """
    Run inference with the ONNX model
    """
    # Get the input name of the ONNX model
    input_name = session.get_inputs()[0].name
    
    # Check if we're using GPU for ONNX
    providers = session.get_providers()
    using_gpu = providers and providers[0] != 'CPUExecutionProvider'
    
    # Prepare input tensor
    if using_gpu and torch.cuda.is_available() and img_tensor.device.type == 'cuda':
        # For GPU inference, we need numpy array, but avoid unnecessary CPU transfer
        img_np = img_tensor.cpu().numpy()
    else:
        # For CPU inference or if tensor is already on CPU
        img_np = img_tensor.cpu().numpy()
    
    # Run inference
    inputs = {input_name: img_np}
    output = session.run(None, inputs)
    return output

def decode_prediction(output, use_onnx=False):
    # This function should be adapted to match your model output format
    # For demonstration purposes, assuming output is logits for character classes
    
    if use_onnx:
        # ONNX output might need reshaping depending on your model
        logits = torch.tensor(output[0])
    else:
        logits = output
    
    # Placeholder for actual decoding logic
    # You'll need to implement this based on your model's output format
    # Example: for classification with 10 digits and 4 positions
    predictions = []
    for i in range(logits.shape[1]):  # Assuming shape is [batch_size, num_positions, num_classes]
        pred = torch.argmax(logits[0, i]).item()
        predictions.append(pred)
    
    # Convert predictions to string  use lst_to_str
    # return lst_to_str(predictions)
    return list_to_str(predictions)
    

def main():
        
    args = parse_args()
    
    args.image_path =  'dataset_real_pre/2bh.1.png'

    args.pytorch_model_path = './checkpoint/0.39_model_resnet.pth'
    args.onnx_model_path = 'checkpoint/0.39_model_resnet.onnx'
    # Load the image
    img_tensor = preprocess_image(args.image_path)
    
    # Load PyTorch model
    print(f"Loading PyTorch model from: {args.pytorch_model_path}")
    torch_model = load_pytorch_model(args.pytorch_model_path, args.model_name, args.device)
    
    # Load ONNX model
    print(f"Loading ONNX model from: {args.onnx_model_path}")
    onnx_session = load_onnx_model(args.onnx_model_path, force_cpu=args.force_onnx_cpu)
    
    # Warmup runs
    print("Performing warmup runs...")
    _ = run_pytorch_inference(torch_model, img_tensor, args.device)
    _ = run_onnx_inference(onnx_session, img_tensor)
    
    # PyTorch inference timing
    print(f"Running PyTorch inference {args.num_runs} times...")
    pytorch_times = []
    for _ in range(args.num_runs):
        start_time = time.time()
        pytorch_output = run_pytorch_inference(torch_model, img_tensor, args.device)
        pytorch_times.append(time.time() - start_time)
    
    # ONNX inference timing
    print(f"Running ONNX inference {args.num_runs} times...")
    onnx_times = []
    for _ in range(args.num_runs):
        start_time = time.time()
        onnx_output = run_onnx_inference(onnx_session, img_tensor)
        onnx_times.append(time.time() - start_time)
    
    # Calculate statistics
    pytorch_avg = np.mean(pytorch_times) * 1000  # Convert to ms
    pytorch_std = np.std(pytorch_times) * 1000
    onnx_avg = np.mean(onnx_times) * 1000
    onnx_std = np.std(onnx_times) * 1000
    
    # Print results
    print("\nPerformance Comparison:")
    print(f"PyTorch ({args.device}): {pytorch_avg:.2f} ± {pytorch_std:.2f} ms per inference")
    print(f"ONNX ({onnx_session.get_providers()[0].split('ExecutionProvider')[0]}): {onnx_avg:.2f} ± {onnx_std:.2f} ms per inference")
    print(f"Speedup: {pytorch_avg / onnx_avg:.2f}x")
    
    # Run a single inference and show the result
    print("\nSample Inference Result:")
    pytorch_result = decode_prediction(pytorch_output)
    onnx_result = decode_prediction(onnx_output, use_onnx=True)
    
    print(f"PyTorch prediction: {pytorch_result}")
    print(f"ONNX prediction: {onnx_result}")
    print(f"Match: {pytorch_result == onnx_result}")

def onnx_inference(image_path, model_path, force_cpu=False):
    # Load the image
    img_tensor = preprocess_image(image_path)
    
    # Load ONNX model
    onnx_session = load_onnx_model(model_path, force_cpu=force_cpu)
    
    # Run inference
    output = run_onnx_inference(onnx_session, img_tensor)
    
    # Decode prediction
    result = decode_prediction(output, use_onnx=True)
    
    return result


def pytorch_inference(image_path, model_path, model_name, device):
    # Load the image
    img_tensor = preprocess_image(image_path)
    
    # Load PyTorch model
    torch_model = load_pytorch_model(model_path, model_name, device)
    
    # Run inference
    output = run_pytorch_inference(torch_model, img_tensor, device)
    
    # Decode prediction
    result = decode_prediction(output, use_onnx=False)
    
    return result

if __name__ == "__main__":
    image_real_folder_path = 'dataset_real/val'
    image_real_files = os.listdir(image_real_folder_path)
    
    model_path = 'lightning_logs/version_19/val_acc=0.46_best_val_acc-epoch=06.ckpt'
    for image_real_file in image_real_files:
        image_path = os.path.join(image_real_folder_path, image_real_file)
        # result = onnx_inference(image_path, 'checkpoint/0.39_model_resnet.onnx', force_cpu=False)
        
        # use pytorch model in cuda 
        result =   pytorch_inference(image_path, model_path, 'resnet', 'cuda')
        
        # true label
        true_label = image_real_file.split('.')[0]
        if len(true_label) < 5:
            true_label = true_label + '_'
        
        # Only print if there's a match
        if result == true_label:
            print("************************************************")
            print(f"True ---- Label: {true_label}, Pred: {result}")
            
        else:
            print(f"False ---- Label: {true_label}, Pred: {result}")
        