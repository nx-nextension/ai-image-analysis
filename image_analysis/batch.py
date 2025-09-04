from lmdeploy import pipeline, PytorchEngineConfig, VisionConfig
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.utils import encode_image_base64
from utils.markdown import extract_json_objects
from typing import List
from urllib.parse import quote_plus, urlparse, parse_qs, unquote
import datetime
import argparse
import json
import os
import sys
import csv
import time

# Please set tp=2 for the 38B version and tp=8 for the 241B-A28B version.
model = "OpenGVLab/InternVL3_5-8B"
pipe = pipeline(
    model,
    backend_config=PytorchEngineConfig(session_len=8192, tp=1),
    vision_config=VisionConfig(max_batch_size=4),
)

# backend_config=TurbomindEngineConfig(
#     max_batch_size=32,
#     enable_prefix_caching=True,
#     cache_max_entry_count=0.8,
#     session_len=8192,
# ))

screenshot_format = {
    'type': 'object',
    'properties': {
        'filter_category': { 'type': 'string' },
        'title': { 'type': 'string' },
        'summary': { 'type': 'string' },
        'keywords': { 'type': 'string' },
        'dewey_classification': { 'type': 'array', 'items': { 'type': 'string' } },        
    },
    'required': ['filter_category']
}
gen_config = GenerationConfig(response_format=dict(type='json_schema', json_schema=dict(name='screenshot', schema=screenshot_format)))

DEFAULT_PROMPT = """Extract content about the screenshot provided in JSON format.
Always follow the rules:
- be extra precise when reading company and product names

JSON Fields:
- filter_category: ok, parked or newly registered domain or placeholder pages or page from domain registrars, pornographic or adult-only content (including sex toys and sex shops), 404. Use values: placeholder | adult | valid
- title: translated to english
- summary (english)
- keywords (english) 
- dewey_classification  
"""

def get_filename(url: str):
    parsed = urlparse(url)
    fragment = parsed.fragment
    params = parse_qs(fragment)
    return f'{quote_plus(params.get("filename", [url])[0])}.json'

def process_batch(batch_number:int, urls: List[str], prompt = DEFAULT_PROMPT, output_dir = './'):
    prompts = [
        (
            prompt,
            load_image(img_url),
        )
        for img_url in urls
    ]
    #response = pipe(prompts, gen_config=gen_config)
    response = pipe(prompts)    
    #response = [{"text": f'```json\n{json.dumps(dict({"id":f"test","filter_category":"valid","title":"testtitle","keywords":"keywords","dewey_classification":"100"}))}```'}]
        
    result = []
    for i, res in enumerate(response):
        if isinstance(res, dict):
            # for testing
            data = extract_json_objects(res['text'])[0]
        else: 
            try:      
                data = extract_json_objects(res.text)[0]
            except Exception as e:
                print(f'error processing url=${urls[i]}')
                print(res)
                print(e)
                data = dict()
                
        result.append(data)
        data['id'] = urls[i]
        
        #print(json.dumps(data, indent=4))                
        #print(output_dir)
        
        output_file = os.path.join(output_dir, get_filename(data['id']))        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)
            
    return result

def process_urls_in_batches(urls, batch_size, prompt, output_dir):
    """
    Generator that yields (batch_number, processed_data) for each batch.
    """
    for i in range(0, len(urls), batch_size):
        batch_number = i // batch_size + 1
        batch_urls = urls[i:i + batch_size]
        time_start_batch = time.time()
        rows = process_batch(batch_number, batch_urls, prompt, output_dir)
        time_elapsed_batch = time.time() - time_start_batch
        sys.stderr.write(f'batch={i} size={len(batch_urls)} elapsed={time_elapsed_batch}s\n')
        for row in rows:            
            yield row
        
def main():
    parser = argparse.ArgumentParser(
        description="Process a list of URLs from stdin in batches and output CSV to stdout."
    )
    parser.add_argument(
        "--batch-size", "-n",
        type=int,
        required=False,
        default=10,
        help="Number of URLs per batch."
    )
    parser.add_argument(
        "--fields", "-f",
        type=str,
        required=False,
        help="Comma-separated list of field names for the CSV header",
        default="filter_category,title,summary,keywords,dewey_classification,id"        
    )
    parser.add_argument(
        "--output-dir", "-d",
        type=str,
        help="Output directory for JSON files",
        default="./"
    )
    parser.add_argument(
        '--prompt','-p', 
        type=str, 
        required=False, 
        default=DEFAULT_PROMPT, 
        help="Prompt to use for analysis. Example: --prompt $'Line1\\nLine2' or use a heredoc.")
    
    args = parser.parse_args()
        
    # read list of image urls from stdin
    urls = [line.strip() for line in sys.stdin if line.strip()]
    sys.stderr.write(f'=> processing {len(urls)} url(s)')
    sys.stderr.flush()
    
    # prepare stdout CSV writer
    fieldnames = [f.strip() for f in args.fields.split(",") if f.strip()]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore", restval="")
    writer.writeheader()
    
    time_start = time.time()
    for row in process_urls_in_batches(urls, args.batch_size, args.prompt, args.output_dir):        
        writer.writerow(row)
        
    time_elapsed = time.time() - time_start    
    sys.stderr.write(f'\n=> TOTAL ELAPSED TIME = {time_elapsed}s ({time_elapsed / len(urls)}/item)\n')
    sys.stderr.flush()
    
            
# cat urls.txt | python3 script.py -n 2 -p $'This is line1\nThis is line2'
# """
# PROMPT=$(cat <<'EOF'
# This is line1
# This is line2
# EOF
# )

# cat urls.txt | python3 script.py -n 2 -p "$PROMPT"
# """            
if __name__ == "__main__":
    main()
    