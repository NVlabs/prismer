import os
import argparse
import torch
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml


from model.prismer_caption import PrismerCaption
from dataset import create_dataset, create_loader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='')
parser.add_argument('--port', default='')

parser.add_argument('--exp_name', default='', type=str)
args = parser.parse_args()

# load config
config = yaml.load(open('configs/caption.yaml', 'r'), Loader=yaml.Loader)['demo']

# generate expert labels
if len(config['experts']) > 0:
    script_name = f'python experts/generate_depth.py'
    os.system(script_name)
    print('***** Generated Depth *****')

    script_name = f'python experts/generate_edge.py'
    os.system(script_name)
    print('***** Generated Edge *****')

    script_name = f'python experts/generate_normal.py'
    os.system(script_name)
    print('***** Generated Surface Normals *****')

    script_name = f'python experts/generate_objdet.py'
    os.system(script_name)
    print('***** Generated Object Detection Labels *****')

    script_name = f'python experts/generate_ocrdet.py'
    os.system(script_name)
    print('***** Generated OCR Detection Labels *****')

    script_name = f'python experts/generate_segmentation.py'
    os.system(script_name)
    print('***** Generated Segmentation Labels *****')

# load datasets
_, test_dataset = create_dataset('caption', config)
test_loader = create_loader(test_dataset, batch_size=1, num_workers=4, train=False)

# load pre-trained model
model = PrismerCaption(config)
state_dict = torch.load(f'logging/caption_{args.exp_name}/pytorch_model.bin', map_location='cuda:0')
model.load_state_dict(state_dict)
tokenizer = model.tokenizer

# inference
model.eval()
with torch.no_grad():
    for step, (experts, data_ids) in enumerate(tqdm(test_loader)):
        captions = model(experts, train=False, prefix=config['prefix'])

        captions = tokenizer(captions, max_length=30, padding='max_length', return_tensors='pt').input_ids
        caption = captions.to(experts['rgb'].device)[0]

        caption = tokenizer.decode(caption, skip_special_tokens=True)
        caption = caption.capitalize() + '.'

        # save caption
        save_path = test_loader.dataset.data_list[data_ids[0]]['image'].replace('jpg', 'txt')
        with open(save_path, 'w') as f:
            f.write(caption)

print('All Done.')
