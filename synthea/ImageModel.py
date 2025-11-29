from collections import deque
import math
import random
import websockets #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import asyncio

from synthea.Config import Config
from synthea.exceptions import InvalidImageDimensionsException

image_server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

class ImageModel:
    """ A class for generating images by connecting to a ComfyUI server instance. """
    def __init__(self):
        self.processing_lock = asyncio.Lock()

    def _parse_dimensions(self, dimensions: str):
        """ 
        Parses a dimension of form '1000x1000' into a width and height, applying the limitations of the
        comfyUI empty latent image node. 
        """
        config: Config = Config()
        values = dimensions.split('x')
        if len(values) != 2:
            raise InvalidImageDimensionsException(f"Couldn't parse {dimensions} into a width and height")
        width, height = int(values[0]), int(values[1])
        if width < 16 or height < 16:
            raise InvalidImageDimensionsException(f"Invalid dimensions {dimensions} - The width and height must be at minimum 16 pixels")
        if width > 16384 or height > 16384:
            raise InvalidImageDimensionsException(f"Invalid dimensions {dimensions} - The width and height must be at most 16384 pixels")
        if width * height > config.image_maximum_pixels:
            raise InvalidImageDimensionsException(
                f"Tried to make image of dimensions {dimensions}, which is larger than the maximum number of pixels {config.image_maximum_pixels}")
        return width, height

    async def generate_image_from_prompt(self, positive_prompt, negative_prompt="", dimensions=""):
        """ Generates an image from a prompt """
        config: Config = Config()
        async with self.processing_lock:
            with open(f'./synthea/comfyui_workflows/{config.image_generation_workflow_name}.json', 'r') as workflow_json:
                seed = random.random() * 1000000000
                workflow: dict = json.load(workflow_json)
                self.apply_values_to_workflow(workflow, positive_prompt, negative_prompt, seed, dimensions)
                images = await self._get_images_from_workflow(workflow)
                print(f"Received {len(images)} images")
                return images

    def apply_values_to_workflow(self, workflow: dict[str, dict[str,str]], positive_prompt: str, negative_prompt: str, seed: int, dimensions: str=""):
        """ Substitutes in values into the workflow """
        config: Config = Config()
        for node in workflow.values():
            if node['_meta']['title'].startswith("Positive Prompt"):
                node["inputs"]['text'] = positive_prompt
            if node['_meta']['title'].startswith("Negative Prompt"):
                node["inputs"]['text'] = negative_prompt
            if node['_meta']['title'].startswith("Sampler"):
                node["inputs"]['seed'] = seed
            if node['_meta']['title'].startswith("Base Image"):
                if dimensions:
                    width, height = self._parse_dimensions(dimensions)
                else:
                    width, height = self._parse_dimensions(config.image_default_dimensions)
                node["inputs"]['width'] = width
                node["inputs"]['height'] = height

    # queues a prompt for generation.
    def _queue_prompt(self, prompt: dict[str, str]) -> dict[str, str]:
        p = {"prompt": prompt, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req =  urllib.request.Request(f"http://{image_server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def _get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{image_server_address}/view?{url_values}") as response:
            return response.read()

    def _get_history(self, prompt_id):
        with urllib.request.urlopen(f"http://{image_server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    async def _get_images_from_workflow(self, workflow: dict[str, str]) -> dict[str, bytes]:
        # client_id = str(id(workflow))  # Generate unique client ID for this request
        async with websockets.connect(f"ws://{image_server_address}/ws?clientId={client_id}") as websocket:
            prompt_id = self._queue_prompt(workflow)['prompt_id']
            output_images = {}

            # tell the 
            while True:
                out = await asyncio.wait_for(websocket.recv(), timeout=300)
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            break #Execution is done
                else:
                    # If you want to be able to decode the binary stream for latent previews, here is how you can do it:
                    # bytesIO = BytesIO(out[8:])
                    # preview_image = Image.open(bytesIO) # This is your preview in PIL image format, store it in a global
                    print(f"Found non-str message!")
                    continue # previews are binary data

            history = self._get_history(prompt_id)[prompt_id]
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                images_output = []
                if 'images' in node_output:
                    for image in node_output['images']:
                        image_data = self._get_image(image['filename'], image['subfolder'], image['type'])
                        images_output.append(image_data)
                output_images[node_id] = images_output

            print("Finished generating images")
            return output_images
    
