import requests
import json
import base64 

class Ollama:
    '''Connect to the Ollama API

    Args:
        url (str): The url of the LibreTranslate endpoint.
        api_key (str): The API key.

    '''
    DEFAULT_URL = 'http://localhost:11434/api/generate'
    DEFAULT_MODEL = 'llava:7b-v1.6'

    def __init__(self, url=None, model=None, api_key=None):
        self.url = OllamaAPI.DEFAULT_URL if url is None else url
        self.model = OllamaAPI.DEFAULT_MODEL if model is None else model
        self.api_key = api_key
        self.prompt = ''
        self.images = []
        self.context = []
        self.is_running = False
        self.parameters = {}
        self.options = {}
        self.set_paramaters()


    def set_paramaters(self, raw: bool = False, stream: bool = False):
        '''
        https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
        '''
        self.parameters['model'] = self.model
        self.parameters['raw'] = raw
        self.parameters['stream'] = stream
        self.set_options()
        return self.parameters


    def set_options(self, num_keep: int = None, seed: int = None, num_predict: int = None, top_k: int = None, top_p: float = None, tfs_z: float = None, typical_p: float = None, repeat_last_n: int = None, temperature: float = None, repeat_penalty: float = None, presence_penalty: float = None, frequency_penalty: float = None, mirostat: int = None, mirostat_tau: float = None, mirostat_eta: float = None, penalize_newline: bool = None, stop: list = None, numa: bool = None, num_ctx: int = None, num_batch: int = None, num_gqa: int = None, num_gpu: int = None, main_gpu: int = None, low_vram: bool = None, f16_kv: bool = None, vocab_only: bool = None, use_mmap: bool = None, use_mlock: bool = None, rope_frequency_base: float = None, rope_frequency_scale: float = None, num_thread: int = None):
        '''
        https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
        '''
        params = locals()
        del params['self']
        self.options.update({key: value for key, value in params.items() if value is not None})
        self.options = {key: value for key, value in self.options.items() if value is not None}
        self.parameters['options'] = self.options
        return self.options


    def update_prompt(self, prompt):
        self.parameters['prompt'] = prompt


    def update_context(self, context):
        self.parameters['context'] = context


    def update_images(self, images):
        if 'images' in self.parameters and images == None:
           del self.parameters['images']
        else:
            self.parameters['images'] = self.encode_images(images)


    def api_request(self, prompt: str, images = None, context: bool = False):
        self.update_prompt(prompt)
        self.update_images(images)
        request_json = requests.post(self.url, json=self.parameters)
        self.get_statuts_code(request_json)
        if self.is_running is True:
            request = json.loads(request_json.text)
            if context is True:
                self.get_context(request)     
            return request


    def get_context(self, request):
        context = request['context']
        self.update_context(context)
        return context


    def get_response(self, request):
        response = request['response']
        return response


    def get_statuts_code(self, request):
        if request.status_code == 200:
            self.is_running = True
        else:
            print(f'Error {request.status_code}: {request.text}')
        return request.status_code


    def encode_images(self, images):
        encoded_images = []
        for image in images:
            encoded_str = base64.b64encode(image).decode('utf-8')
            encoded_images.append(encoded_str)
        return encoded_images

'''
ollama_instance = OllamaAPI()
list0 = []
with open(r'test.jpg', 'rb') as image_file:
    image_content = image_file.read()
    list0.append(image_content)

while True :
    prompt=str((input('Me: ')))
    request = ollama_instance.api_request(prompt = prompt, images = list0)
    response = ollama_instance.get_response(request = request)
    print(response)
'''
