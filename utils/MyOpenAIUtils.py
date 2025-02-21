from openai import OpenAI
from pydantic import BaseModel

OPENAI_CLIENT = OpenAI()


def get_openai_inference(query):
    completion = OPENAI_CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": query}
        ]
    )

    return completion.choices[0].message.content


def get_openai_inference_with_schema(query, requested_schema):
    completion = OPENAI_CLIENT.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": query}
        ],
        response_format=requested_schema,
    )

    return completion.choices[0].message.parsed


_GPT_MODEL = "gpt-4o-mini"
