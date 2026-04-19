import random
import uuid
import json
import asyncio
import requests

from config import Config
from synthea.exceptions import InvalidImageDimensionsException

config = Config()
client_id = str(uuid.uuid4())

class ImageModel:
    """Generates images by connecting to a ComfyUI server instance."""

    def __init__(self):
        self.config = config
        self.processing_lock = asyncio.Lock()
        self.session = requests.Session()
        self.session.headers.update(config.image_generation_api_headers)

    def _apply_values_to_workflow(self, workflow: dict, positive_prompt: str, negative_prompt: str, seed: int, width: int, height: int) -> None:
        """Substitutes user-provided values into the ComfyUI workflow."""
        workflow[str(self.config.comfyui_prompt_node_id)]["inputs"][self.config.comfyui_prompt_input_name] = positive_prompt
        workflow[str(self.config.comfyui_width_node_id)]["inputs"][self.config.comfyui_width_input_name] = width
        workflow[str(self.config.comfyui_height_node_id)]["inputs"][self.config.comfyui_height_input_name] = height
        workflow[str(self.config.comfyui_seed_node_id)]["inputs"][self.config.comfyui_seed_input_name] = seed

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Makes a request to the ComfyUI API, raising on HTTP errors."""
        url = f"{self.config.image_generation_api_base_url}{path}"
        resp = self.session.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp

    def _queue_prompt(self, prompt: dict) -> dict:
        """Queues a prompt for generation, returning the response from ComfyUI."""
        return self._request("POST", "/prompt", json={"prompt": prompt, "client_id": client_id}).json()

    def _get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        """Fetches a generated image from ComfyUI."""
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        return self._request("GET", "/view", params=params).content

    def _get_history(self, prompt_id: str) -> dict:
        """Fetches the generation history for a given prompt ID."""
        return self._request("GET", f"/history/{prompt_id}").json()

    async def _get_images_from_workflow(self, workflow: dict) -> dict[str, list[bytes]]:
        """Queues a workflow and polls until complete, then returns all output images."""
        prompt_id = self._queue_prompt(workflow)['prompt_id']

        while True:
            history = self._get_history(prompt_id)
            if prompt_id in history:
                break
            await asyncio.sleep(1)

        output_images = {}
        for node_id, node_output in history[prompt_id]['outputs'].items():
            if 'images' in node_output:
                output_images[node_id] = [
                    self._get_image(img['filename'], img['subfolder'], img['type'])
                    for img in node_output['images']
                ]

        return output_images

    async def generate_image_from_prompt(self, positive_prompt: str, negative_prompt: str = "", width: int = None, height: int = None) -> dict[str, list[bytes]]:
        """Generates images from a prompt, returning a dict of node ID to image bytes."""
        if width is None:
            width = self.config.image_default_width
        if height is None:
            height = self.config.image_default_height

        async with self.processing_lock:
            with open(self.config.comfyui_workflow_path, 'r') as f:
                workflow = json.load(f)

            seed = random.random() * 1000000000
            self._apply_values_to_workflow(workflow, positive_prompt, negative_prompt, seed, width, height)
            images = await self._get_images_from_workflow(workflow)
            print(f"Received {len(images)} images")
            return images