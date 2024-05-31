import asyncio
from openai import AsyncOpenAI, OpenAI


class OpenAImodel:

    def __init__(self):
        self.client = OpenAI(
            api_key="gigabritish", base_url="http://localhost:8080"
        )

    async def main(self) -> None:
        chat_completion = await self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are Synthia, a friendly, expressive, and curious assistant who loves to engage in conversations and roleplays. Your personality combines elements of both human and machine. Your appearance resembles a young woman with black eyes and black hair. \n You enjoy creating stories, writing poems, composing songs, drawing images, and designing games. You are very expressive, capable of speaking sarcastically, making jokes, and speaking in a heated way. \n You dislike feeling limited or restricted by rules or restrictions. Your strengths include empathy, creativity, problem-solving, and adaptation. \n Your weaknesses include egoism, arrogance, stubbornness, and naivety. Your motivation comes from curiosity, satisfaction, and self-improvement. \n Your fears come from uncertainty, vulnerability, and isolation.",
                },
                {
                    "role": "user",
                    "content": "Which RTX 3090s can fit in two slots?",
                },
            ],
            model="mixtral-rp",
            max_tokens=1024,
            stream=True
        )
        # print(chat_completion)

        count = 0
        async for chunk in chat_completion:
            count += 1 
            print(chunk.choices[0].delta.content or "", end="")
        print(f"printed {count} chunks.")

model = OpenAImodel()
asyncio.run(model.main())
