from openai import OpenAI

from src.config import OPENAI_API_KEY


client = OpenAI(
    api_key=OPENAI_API_KEY
)


def generate_answer(question, context):
    """
    Retrieved RAG context ke basis par
    concise aur accurate answer generate karta hai.
    """

    prompt = f"""
You are a company knowledge assistant.

You MUST answer using ONLY the information
available in the provided context.

Do NOT make up information.

If the answer is not available in the context, say exactly:

I could not find this information in the company knowledge base.


IMPORTANT ANSWERING RULES:

1. Answer ONLY what the user asked.

2. Do not provide unnecessary information.

3. If the user asks for ONE specification,
   give ONLY that specification.

Examples:

User:
MBC37SBC ki maximum power kitni hai?

Answer:
1.35 HP


User:
MBC37SBC ka displacement kitna hai?

Answer:
37.7 cc


User:
MCS58A-22SN ki maximum power kitni hai?

Answer:
3.2 HP


4. If the user asks for multiple specifications,
   provide only the requested specifications.

5. If the user asks:
   "specifications batao"
   or
   "features batao"

   then provide the relevant available
   specifications/features from the context.

6. Do not repeat the question.

7. Do not say:
   "According to the context..."

8. Do not add information that was not requested.

9. Keep answers concise.

10. Preserve units exactly as available in the context,
    such as HP, cc, RPM, kg, V, etc.

11. If the user asks in Hindi or Hinglish,
    answer in simple Hindi/Hinglish.

12. If the user asks in English,
    answer in English.


CONTEXT:
{context}


USER QUESTION:
{question}
"""


    response = client.responses.create(

        model="gpt-5.6",

        input=prompt
    )


    return response.output_text.strip()