import gradio as gr
import torch
import os
import json
import pandas as pd

###
def count_layers(model_state):
    layer_types = {'weight': 0, 'bias': 0, 'running_mean': 0, 'running_var': 0}
    total_parameters = 0
    for key, tensor in model_state.items():
        if torch.is_tensor(tensor):
            total_parameters += torch.numel(tensor)
            if 'weight' in key:
                layer_types['weight'] += 1
            elif 'bias' in key:
                layer_types['bias'] += 1
            elif 'running_mean' in key:
                layer_types['running_mean'] += 1
            elif 'running_var' in key:
                layer_types['running_var'] += 1
    return layer_types, total_parameters

#
def infer_architecture(layer_names):
    if any("res" in name for name in layer_names):
        return "ResNet"
    elif any("dw" in name for name in layer_names):
        return "MobileNet"
    elif any("efficient" in name for name in layer_names):
        return "EfficientNet"
    else:
        return "Unknown or other"

#
def process_pth(file):
    df = pd.DataFrame(columns=['Pth File', 'Information', 'Layer Counts', 'Total Parameters', 'File Size', 'Inferred Architecture'])
    try:
        model_state = torch.load(file.name, map_location='cpu')
        if 'model' in model_state:
            model_state = model_state['model']

        #
        layer_counts, total_parameters = count_layers(model_state)

        #
        inferred_architecture = infer_architecture(model_state.keys())

        #
        main_layers = [k for k in model_state.keys() if not any(sub in k for sub in ['bias', 'running_mean', 'running_var'])]
        total_main_layers = len(main_layers)

        #
        first_tensor_key = list(model_state.keys())[0]
        last_tensor_key = list(model_state.keys())[-1]
        first_tensor = model_state[first_tensor_key]
        last_tensor = model_state[last_tensor_key]

        first_layer_shape = list(first_tensor.shape) if torch.is_tensor(first_tensor) else "Unknown"
        last_layer_shape = list(last_tensor.shape) if torch.is_tensor(last_tensor) else "Unknown"

        #
        info = {
            'Layer Count': layer_counts,
            'Total Parameters': total_parameters,
            'File Size (KB)': os.path.getsize(file.name) // 1024,
            'Total Main Layers': total_main_layers,
            'First Layer Shape': first_layer_shape,
            'Last Layer Shape': last_layer_shape,
            'Inferred Architecture': inferred_architecture
        }

        return json.dumps(info, indent=4)
    except Exception as e:
        return f"Failed to process the file: {e}"

#Gradio
iface = gr.Interface(
    fn=process_pth,
    inputs=gr.File(label="Upload .PTH File"),
    outputs="text",
    title="PTH-Scope",
    description="Upload a .PTH file to analyze its structure and parameters. A .PTH file is typically a PyTorch model file, which contains the state of a neural network trained using libraries like PyTorch. These files encapsulate the learned weights and biases of the model after training. ",
    examples=[["mobilenetv2_035.pth"]]
)


#
iface.launch(debug=True)
