import unittest
from dotenv import load_dotenv
from openai import OpenAI
from unittest.mock import patch, MagicMock
from simple_model_api import SimpleModelAPI

class TestSimpleModelAPI(unittest.TestCase):

    @patch('simple_model_api.OpenAI')
    def setUp(self, MockOpenAI):
        self.mock_client = MockOpenAI.return_value
        self.model_api = SimpleModelAPI("gpt-3.5-turbo")

    def test_initialization(self):
        self.assertEqual(self.model_api.model_name, "gpt-3.5-turbo")
        self.assertEqual(self.model_api.conversation, [])
        self.assertIsNotNone(self.model_api.client)

    @patch('simple_model_api.OpenAI')
    def test_call_with_empty_conversation(self, MockOpenAI):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Mocked response"
        self.mock_client.chat.completions.create.return_value = mock_response

        response = self.model_api.call("What is on the menu?")
        self.assertEqual(response, "Mocked response")
        self.assertEqual(len(self.model_api.conversation), 2)

    @patch('simple_model_api.OpenAI')
    def test_call_with_existing_conversation(self, MockOpenAI):
        self.model_api.conversation = [
            {"role": "system", "content": self.model_api._prompt},
            {"role": "user", "content": "What is on the menu?"}
        ]
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Mocked response"
        self.mock_client.chat.completions.create.return_value = mock_response

        response = self.model_api.call("Can I have a Margherita Pizza?")
        self.assertEqual(response, "Mocked response")
        self.assertEqual(len(self.model_api.conversation), 3)

    @patch('simple_model_api.OpenAI')
    def test_call_with_exception(self, MockOpenAI):
        self.mock_client.chat.completions.create.side_effect = Exception("API error")

        response = self.model_api.call("What is on the menu?")
        self.assertEqual(response, "An error occurred while calling the model")

def eval_vs_ideal(test_set, assistant_answer):
    """
    Evaluate the assistant's response against the ideal answer using an OpenAI model
    """

    cust_msg = test_set['customer_msg']
    ideal = test_set['ideal_answer']
    completion = assistant_answer

    system_message = """\
    You are an assistant that evaluates how well the customer service agent \
    answers a user question by comparing the response to the ideal (expert) response
    Output a single letter and nothing else. 
    """

    user_message = f"""\
You are comparing a submitted answer to an expert answer on a given question. Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {cust_msg}
    ************
    [Expert]: {ideal}
    ************
    [Submission]: {completion}
    ************
    [END DATA]

Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.
    The submitted answer may either be a subset or superset of the expert answer, or it may conflict with it. Determine which case applies. Answer the question by selecting one of the following options:
    (A) The submitted answer is a subset of the expert answer and is fully consistent with it.
    (B) The submitted answer is a superset of the expert answer and is fully consistent with it.
    (C) The submitted answer contains all the same details as the expert answer.
    (D) There is a disagreement between the submitted answer and the expert answer.
    (E) The answers differ, but these differences don't matter from the perspective of factuality.
  choice_strings: ABCDE
"""

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
        )
    return response

class TestLLMSimpleModelAPI(unittest.TestCase):
    """
    Test the SimpleModelAPI class with the GPT to evaluate the assistant's response
    """
    def setUp(self) -> None:
        self.model_api = SimpleModelAPI("gpt-3.5-turbo")

    def test_simple_question(self):
        test_set = {
            'customer_msg': "What is on the menu?",
            'ideal_answer': ("The menu is composed of the following items: "
                             "Margherita Pizza, Pepperoni Pizza, Hawaiian Pizza.")
        }
        assistant_answer = self.model_api.call(test_set['customer_msg'])
        response = eval_vs_ideal(test_set, assistant_answer)
        self.assertEqual(response.choices[0].message.content, "A",
                         (f"Ideal answer: {test_set['ideal_answer']}, "
                          f"Assistant answer: {assistant_answer}")
                         )


if __name__ == '__main__':
    load_dotenv()
    unittest.main()
