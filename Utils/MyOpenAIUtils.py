from openai import OpenAI

OPENAI_CLIENT = OpenAI()

def get_openai_inference(query):
    completion = OPENAI_CLIENT.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "user", "content": query}
      ]
    )

    return completion.choices[0].message.content

