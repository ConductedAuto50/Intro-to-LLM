from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


llm=ChatGoogleGenerativeAI(model="gemini-1.0-pro")

fewshotprompt= '''Let's solve some linear equations in two variables! Here are some examples:

Let x be price of pen and y be price of notebook.

Ex1: Sam buys 1 pens and 1 notebooks for $7. Mark buys 2 pen and 3 notebooks for $18.
A1: the equations will be x+y=7 and 2x+3y=18. therefore the solution is x=3 and y=4.

Ex2: Sam buys 3 pens and 2 notebooks for $10.5. Mark buys 1 pen and 4 notebooks for $13.5.
A2: the equations will be 3x+2y=10.5 and x+4y=13.5. therefore the solution is x=1.5 and y=3.

Ex2: Sam buys 3 pens and 1 notebooks for $6. Mark buys 1 pen and 2 notebooks for $6.
A2: the equations will be 3x+y=6 and x+2y=6. therefore the solution is x=1.2 and y=2.4.

Now solve the following:
Sam buys 3 pens and 2 notebooks for $15. Mark buys 1 pen and 4 notebooks for $12. What is the price of one pen and one notebook?


'''

class Answer(BaseModel):
    cost_of_pen: float = Field(description="cost of a pen ($) in decimals (x)")
    cost_of_notebook: float = Field(description="cost of a notebook ($) in decimals (y)")


parser = JsonOutputParser(pydantic_object=Answer)

prompt = PromptTemplate(
    template="Answer the following: \n{query}\n{format_instructions}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser
resp=chain.invoke({'query':fewshotprompt})
print(resp)
