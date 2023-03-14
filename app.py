import gradio as gr
from gradio.components import Image as grImage
from gradio.components import Textbox as Textbox
from model.prismer_vqa import PrismerVQA
from model.prismer_caption import PrismerCaption
from dataset.utils import *
from pathlib import Path
from dataset import create_dataset, create_loader
import fire
import multiprocessing
import logging

transform = Transform(resize_resolution=480, scale_size=[
    0.5, 1.0], train=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def call_expert_scripts(config):
    if len(config['experts']) > 0:
        script_names = ["python experts/generate_depth.py",
                        "python experts/generate_edge.py",
                        "python experts/generate_normal.py",
                        "python experts/generate_objdet.py",
                        "python experts/generate_ocrdet.py",
                        "python experts/generate_segmentation.py"]
        # If you have constrained resources, run the scripts sequentially:
        # for script_name in script_names:
        #     os.system(script_name)
        with multiprocessing.Pool(6) as p:
            p.map(os.system, script_names)


def get_model_input(config):
    _, test_dataset = create_dataset('caption', config)
    if len(test_dataset) != 1:
        logging.warning(
            "Make sure to empty the helpers/images folder before running the demo. Otherwise you are recomputing for images that aren't shown ")
    test_loader = create_loader(
        test_dataset, batch_size=1, num_workers=4, train=False
    )
    experts, _ = next(iter(test_loader))
    return experts


def prepare_inputs(image, question, task="caption"):
    image = transform(image, None)
    image['rgb'] = image['rgb'].unsqueeze(0).to(device)
    if task == "caption":
        question = pre_caption(question, max_words=50)
    else:
        question = pre_question(question, max_words=50)
    return image, question


def move_to_device(experts):
    for key in experts:
        if key == 'obj_detection':
            experts[key]['label'] = experts[key]['label'].to(device)
            experts[key]['instance'] = experts[key]['instance'].to(device)
        else:
            experts[key] = experts[key].to(device)
    return experts


def demo(task: str = "vqa", model_name: str = "prismerz_base"):
    use_experts = "prismerz" not in model_name
    if use_experts:
        data_path = Path("helpers")
        label_path = data_path / "labels"
        label_path.mkdir(exist_ok=True, parents=True)
    config = {
        "prismer_model": model_name.replace("z", ""),
        "experts": ["none"] if not use_experts else ['depth', 'normal', 'seg_coco', 'edge', 'obj_detection', 'ocr_detection'],
        "data_path": data_path if use_experts else None,
        "label_path": label_path if use_experts else None,
        "freeze": "freeze_vision",
        "image_resolution": 480,
        "prefix": "" if task == "vqa" else "A picture of",
        "dataset": "demo"
    }
    if task == "vqa":
        model = PrismerVQA(config)
    elif task == "caption":
        model = PrismerCaption(config)
    else:
        raise ValueError(f"Task {task} not supported")
    state_dict = torch.load(
        f'logging/{task}_{model_name}/pytorch_model.bin', map_location=device
    )
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    img_path = config["data_path"] / "images" / "image.jpg"

    def infer(image, question):
        if use_experts:
            image.save(img_path)
            call_expert_scripts(config)
            experts = get_model_input(config)

        im, question = prepare_inputs(image, question, task)
        experts = im if not use_experts else experts
        experts = move_to_device(experts)

        with torch.no_grad():
            if task == "caption":
                answer = model(experts, prefix=question, train=False, inference='generate')
            else:
                answer = model(experts, [question], train=False, inference='generate')

        if use_experts:
            img_path.unlink()
            expert_images = [label_path / key / "helpers" / "images" / "image.png" for key in experts]
            expert_images = [str(im_path) for im_path in expert_images if img_path.exists()]
            labels = ["Depth", "Normal", "Segmentation", "Edge", "Object Detection", "OCR Detection"]
            outs = list(zip(expert_images, labels))
            return answer[0], outs
        else:
            return answer[0]

    # Prepare the interface
    model_title = "PrismerZ" if "prismerz" in model_name else "Prismer"
    model_size = model_name.split("_")[-1].capitalize()
    title = f"{model_title} {model_size} {task.capitalize()}"
    if task == "vqa":
        inputs = [grImage(type="pil"), Textbox(placeholder="What is the color of the shirt?")]
        description = f"Visual Question Answering with {model_title} {model_size}"
        examples = [
            ["http://images.cocodataset.org/val2017/000000039769.jpg",
             "How many cats are there?"],
            ["https://ids.si.edu/ids/deliveryService?max_w=800&id=NPG-NPG_2001_13",
             "What is the man holding in his hand?"],
        ]
    else:
        inputs = [grImage(type="pil"), Textbox(placeholder="A picture of", interactive=False)]
        description = f"Image Captioning with {model_title} {model_size}"
        examples = [
            ["http://images.cocodataset.org/val2017/000000039769.jpg", "A picture of"],
            ["https://ids.si.edu/ids/deliveryService?max_w=800&id=NPG-NPG_2001_13", "A picture of"],
        ]

    if use_experts:
        outputs = [Textbox(placeholder="Answer", interactive=False),
                   gr.Gallery(label="Experts")]
    else:
        outputs = Textbox(placeholder="Answer", interactive=False)
    gr.Interface(infer, inputs, outputs, title=title, description=description, examples=examples).launch()


if __name__ == "__main__":
    fire.Fire(demo)
