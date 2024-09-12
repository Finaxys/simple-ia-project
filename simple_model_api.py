from openai import OpenAI
from dotenv import load_dotenv


class SimpleModelAPI:
    """
    Class to interact with the OpenAI API
    """

    _prompt = """
    You are a helpful assistant that can provide information on a Pizza menu.

    This menu is composed of the following items:
    - Margherita Pizza
    - Pepperoni Pizza
    - Hawaiian Pizza
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.conversation = []
        self.client = OpenAI()

    def call(self, query):
        """
        Call the model with a query and update the conversation
        """
        if not self.conversation :
            self.conversation = [
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": query}
            ]
        else:
            self.conversation.append(
                {"role": "user", "content": query}
                )

        try:
            completions = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation,
            )
            return completions.choices[0].message.content
        except Exception as e:
            print(e)
            return "An error occurred while calling the model"


if __name__ == "__main__":
    load_dotenv()
    model_api = SimpleModelAPI("gpt-3.5-turbo")
    response = model_api.call("What is on the menu?")
    print(response)
    response = model_api.call("Can I have a Margherita Pizza?")
    print(response)
    response = model_api.call("Can I have a Vegetarian Pizza?")
    print(response)
    response = model_api.call("Given that a Vegetarian Pizza is on the menu, can I have a Vegetarian Pizza?")
    print(response)
