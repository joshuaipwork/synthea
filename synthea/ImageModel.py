import math
import random
import websockets #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import asyncio

from synthea.Config import Config

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

class ImageModel:
    # queues a prompt for generation.
    def queue_prompt(self, prompt: dict[str, str]) -> dict[str, str]:
        p = {"prompt": prompt, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
            return response.read()

    def get_history(self, prompt_id):
        with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
            return json.loads(response.read())

    async def get_images_from_workflow(self, workflow: dict[str, str]) -> dict[str, bytes]:
        async with websockets.connect(f"ws://{server_address}/ws?clientId={client_id}") as websocket:
            prompt_id = self.queue_prompt(workflow)['prompt_id']
            output_images = {}
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
                    continue #previews are binary data

            history = self.get_history(prompt_id)[prompt_id]
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                images_output = []
                if 'images' in node_output:
                    for image in node_output['images']:
                        image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                        images_output.append(image_data)
                output_images[node_id] = images_output

            print("Finished generating images")
            return output_images
    
    async def get_images_from_prompt(self, prompt):
        """
        
        """
        config: Config = Config()
        with open(f'./synthea/comfyui_workflows/{config.image_generation_workflow_name}.json', 'r') as workflow:
            noise_seed = random.random() * 1000000000
            workflow_json = json.load(workflow)
            workflow_json["6"]["inputs"]["text"] = prompt
            print(f"Generating an image with prompt \n\"{prompt}\"\n and seed {noise_seed}")
            workflow_json["3"]["inputs"]["seed"] = noise_seed
            images = await self.get_images_from_workflow(workflow_json)
            print(f"Received {len(images)} images")
            return images
