import gradio as gr
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T


model_loaded = False
try:
    from fastai.vision.all import *
    model_path = Path("stroke_model.pkl")
    if model_path.exists():
        try:
            learn = load_learner(model_path)
            model_loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Model loading error: {e}")
    else:
        print("Model file not found: stroke_model.pkl")
except Exception as e:
    print(f"Error with FastAI or dependencies: {e}")

# Prediction function
def predict_stroke(img):
    
    if not model_loaded:
        return {
            "Stroke Present": 0.5, 
            "No Stroke": 0.5
        }, "Model could not be loaded. We are in maintenance mode. Please try again later."
    
    try:
        # Process the image
        if img is None:
            return {"Stroke Present": 0.0, "No Stroke": 0.0}, "Error: Image could not be loaded"
        
        # Convert to PIL image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
            
        
        img_array = np.array(img)
        if len(img_array.shape) == 3:  
            if not np.allclose(img_array[:,:,0], img_array[:,:,1]) or not np.allclose(img_array[:,:,1], img_array[:,:,2]):
                return {"Stroke Present": 0.0, "No Stroke": 0.0}, "Warning: This image does not appear to be a brain CT scan. Please upload a valid brain CT scan."
        
        # Apply transformations needed by the model
        transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0)
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = img_tensor.to(device)
        learn.model.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = learn.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get probabilities with fixed indices (by default 0: stroke_present, 1: no_stroke)
        stroke_prob = float(probabilities[0])
        no_stroke_prob = float(probabilities[1])
        
        # Determine which class has higher probability
        predicted_class = "Stroke Present" if stroke_prob > no_stroke_prob else "No Stroke"
        

        results = {
            "Stroke Present": stroke_prob,
            "No Stroke": no_stroke_prob
        }
        
        label = f"Prediction: {predicted_class}"
        return results, label
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {"Stroke Present": 0.0, "No Stroke": 0.0}, f"Error: {str(e)}"


stroke_examples = [
    "10036.png", 
    "10101.png", 
    "109 (8).jpg", 
    "17032.png", 
    "17028.png",  
]

no_stroke_examples = [
    "10083.png",  
    "10086.png", 
    "10087.png", 
    "17033.png" ,
    "17031.png" 
]

# Gradio interface
with gr.Blocks(title="Stroke Detection from CT Scans") as demo:
    gr.Markdown("# Stroke Detection from CT Scans")
    
    if not model_loaded:
        gr.Markdown("### ⚠️ SYSTEM MAINTENANCE MODE ⚠️")
        gr.Markdown("The model is temporarily unavailable. Our technical team is working to fix the issue.")
    
    gr.Markdown("Upload a CT scan image to check for signs of stroke (hemorrhagic or ischemic).")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="CT Scan Image")
            submit_btn = gr.Button("Analyze Image")
        
        with gr.Column():
            label_output = gr.Label(label="Diagnosis")
            result_output = gr.Label(label="Results")
    
    # Use tabs for side-by-side examples
    with gr.Tabs():
        with gr.TabItem("Stroke Cases (Positive Examples)"):
            gr.Examples(
                examples=stroke_examples,
                inputs=input_img
            )
        
        with gr.TabItem("No Stroke Cases (Negative Examples)"):
            gr.Examples(
                examples=no_stroke_examples,
                inputs=input_img
            )
    
    submit_btn.click(
        fn=predict_stroke,
        inputs=input_img,
        outputs=[result_output, label_output]
    )
    
    gr.Markdown("## Important Note")
    gr.Markdown("This is a demonstration tool only and should not be used for actual medical diagnosis. Always consult with a qualified healthcare professional.")


if __name__ == "__main__":
    demo.launch()