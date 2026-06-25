import os
from openai import OpenAI


def openrouter_mutate(prompt: str, temperature: float, model="mistralai/devstral-small:free"): # TODO: removed the first arg (cfg) since its is not used, so make sure it gets called correctly
    """
    Uses the OpenRouter API to generate a mutation.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    try:
        completion = client.chat.completions.create(
            temperature=temperature,
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt 
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenRouter API: {e}")
        return "" # Return an empty string on failure