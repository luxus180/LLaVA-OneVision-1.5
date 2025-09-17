from datasets import load_dataset
from multiprocessing import Pool
from tool import cfg,get_ip_info,get_init_file
import os
from functools import partial
from tqdm import tqdm
import json

def parese_dataset(data_item,ip_indx,ip_num,dst_dir):
    try:
        index, item = data_item
        if index%ip_num!=ip_indx:
            return 
        name=item['id'].replace('/','_')
        name=os.path.splitext(name)[0]
        
        image_path=os.path.join(dst_dir,name+'.jpg')
        item['image'].save(image_path)
        json_data={
            "messages": [
                {
                    "content": "<image>",
                    "role": "user"
                },
                {
                "content": item['caption'],
                "role": "assistant"
            }
            ],
            "images": [
                image_path
            ]
        }
        json_path=os.path.join(dst_dir,name+'.json')
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=None)
        
    except Exception as e:
        print(f'{item['id']} has exeption {e}')

def main(workers):
    data_path=cfg['hf_data']
    TOKEN_INFO_FILE,MAX_TOKEN_LEN,save_files_dir,big_dir,DEFAULT_DIRECTORY=get_init_file()
    dataset = load_dataset(data_path,data_files='*/*.parquet', split="train", streaming=True) 
    data_iter = enumerate(dataset)
    ip_indx,ip_num,_=get_ip_info()
    with Pool(processes=workers) as pool, tqdm(total=8.5e8, desc="copy") as bar:
        for _ in pool.imap_unordered(partial(parese_dataset,ip_indx=ip_indx,ip_num=ip_num,dst_dir=DEFAULT_DIRECTORY), data_iter):
            bar.update()

if __name__=="__main__":
    main(10)