import gradio as gr
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
import cv2
import numpy as np


def prepare_image(img):
    img = Image.fromarray(img)

    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()  # Convert image to tensor
    ])
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    return img


def load_model(model_path):
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model.encoder1.enc1conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.load_state_dict(torch.load(model_path))

    return model


def create_mask_overlay(img, msk):
    assert img.dtype == msk.dtype, f"{img} dtype should be same as {msk} dtype but found {img.dtype} and {msk.dtype}"
    assert msk.shape[-1] == 1, f"Mask channel dimension should be only 1, but found {msk.shape[-1]}"
    mask = cv2.merge([msk, msk, msk])
    mask[:, :, 1:3] = 0
    mask = cv2.resize(mask, (512, 512))

    blended = cv2.addWeighted(img, 0.7, mask, 0.3, 0)

    return blended


def prepare_output_mask(msk):
    out = torch.squeeze(msk, dim=0)
    out = torch.permute(out, dims=[1, 2, 0])
    out = out.detach().numpy()
    out = (out > 0.5).astype(np.uint8)
    out = out * 255

    return out


def predict(img):
    # Loading the model
    model = load_model("model_pytorch.pt")
    # creating numpy
    # Prepare the image for model input
    prep_img = prepare_image(img)
    # Doing inference
    with torch.no_grad():
        out = model(prep_img)
    # created the mask from the output of the model
    mask = prepare_output_mask(out)
    # creating overlaid image (blended image of mask and original image)
    get_overlaid = create_mask_overlay(img, mask)

    return get_overlaid


interface = gr.Interface(
    fn=predict,
    title="Aestheosceloric Plaque Segmentation Using Computer Vision",
    description="This application utilizes a state-of-the-art U-Net model"
                " to perform segmentation of Aestheosceloric plaques in medical images."
                " The U-Net model, a type of convolutional neural network (CNN), is particularly "
                "effective for image segmentation tasks due to its encoder-decoder architecture, "
                "which allows it to capture both local and global features of the image. Upload an image"
                " to get a segmented output highlighting the regions affected by Aestheosceloric plaques. "
                ""
                "This tool aims to assist healthcare professionals in the diagnosis and treatment planning by providing accurate and reliable segmentation results.",
    inputs=["image"],
    outputs=["image"],
)


if __name__ == '__main__':
    interface.launch()
