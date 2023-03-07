import gradio as gr
from gradio.components import Image as grImage
from gradio.components import Textbox as Textbox
from model.prismer_vqa import PrismerVQA
import yaml
from dataset.utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('configs/gradio_vqa.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

model = PrismerVQA(config)
state_dict = torch.load(f'logging/prismerz_vqa/pytorch_model.bin', map_location=device) # , map_location='cuda:0')
model.load_state_dict(state_dict)
tokenizer = model.tokenizer
model.eval()

transform = Transform(resize_resolution=config['image_resolution'], scale_size=[0.5, 1.0], train=False)


def infer(image, question):
    with torch.no_grad():
        image = transform(image, None)
        image['rgb'] = image['rgb'].unsqueeze(0).to(device)
        question = pre_question(question, max_words=50)
        answer = model(image, [question], train=False, inference=config['inference'])
        return answer[0]


inputs = [grImage(type="pil"), Textbox(placeholder="What is the color of the shirt?")]
outputs = Textbox()

title = "PrismerZ VQA"
description = "Visual Question Answering with PrismerZ"
article = ""

examples = [
    ["http://images.cocodataset.org/val2017/000000039769.jpg", "How many cats are there?"],
    ["http://images.cocodataset.org/val2017/000000039769.jpg", "What color is the blanket?"],
]

gr.Interface(infer, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()

