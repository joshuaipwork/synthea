

import asyncio

import torch
import torch.multiprocessing as multiprocessing

from synthea.ExllamaV2Model import ExLlamaV2Model
from synthea.dtos.GenerationRequest import GenerationRequest
from synthea.dtos.ResponseUpdate import ResponseUpdate


class SyntheaServer:

    def __init__(self, request_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue):
        # a queue to send user prompts to SyntheaServer
        self.request_queue = request_queue
        # a queue to receive streamed inferences from SyntheaServer
        self.response_queue = response_queue
        self.model = None

        torch.cuda.empty_cache()
        self.model = ExLlamaV2Model()

    async def run(self):
        while True:
            generation_request: GenerationRequest = await asyncio.to_thread(self.request_queue.get)  # Wait for a message
            print(f"Received message: {generation_request.context}")

            try:
                self.model.begin_stream(generation_request.context)

                # pull the next chunk of text
                eos: bool = False
                response: str = ""
                while not eos:
                    chunk: list[str] = []
                    for i in range(0,200):
                        token, eos, _ = self.model.generator.stream()
                        if eos: break
                        chunk.append(token)

                    # after some tokens are generated, tell client to stream message to user
                    response += "".join(chunk)
                    self.response_queue.put(
                        ResponseUpdate(
                            generation_request.response_index,
                            eos,
                            response
                        )
                    )
                print(response)
            except Exception as error:
                self.response_queue.put(
                    ResponseUpdate(
                        generation_request.response_index,
                        message_is_completed=True,
                        error=error
                    )
                )        

def run_server_for_test(request_queue, response_queue):
    # Synchronous wrapper to run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server: SyntheaServer = SyntheaServer(request_queue, response_queue)
    loop.run_until_complete(server.run())
    loop.close()

async def test_client(request_queue, response_queue):
    while True:
        response_update: ResponseUpdate = await asyncio.to_thread(response_queue.get)
        print(response_update.new_message)
        if response_update.message_is_completed: print("Message completed.")

def run_client_for_test(request_queue, response_queue):
    # Synchronous wrapper to run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(test_client(request_queue, response_queue))
    loop.close()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    request_queue: multiprocessing.Queue = multiprocessing.Queue()
    response_queue: multiprocessing.Queue = multiprocessing.Queue()

    try:
        server_process = multiprocessing.Process(target=run_server_for_test, args=(request_queue, response_queue))
        server_process.start()

        client_process = multiprocessing.Process(target=run_client_for_test, args=(request_queue, response_queue))
        client_process.start()

        while True:

            print()
            instruction = input("User: ")
            print()
            print("Assistant:", end = "")

            request_queue.put(GenerationRequest(1, f"[INST] {instruction} [/INST]"))
    finally:
        server_process.terminate()
        client_process.terminate()
        torch.cuda.empty_cache()
