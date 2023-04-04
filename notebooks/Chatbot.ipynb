{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "e2fd8a30",
   "metadata": {},
   "source": [
    "# Chatbot"
   ]
  },
  {
   "cell_type": "raw",
   "id": "e909286a",
   "metadata": {},
   "source": [
    "Now let’s start with creating an end-to-end chatbot using Python. I’ll start this task by importing the necessary Python libraries for this task:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d1f646be",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\Sanket\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import os\n",
    "import nltk\n",
    "import ssl\n",
    "import streamlit as st\n",
    "import random\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "ssl._create_default_https_context = ssl._create_unverified_context\n",
    "nltk.data.path.append(os.path.abspath(\"nltk_data\"))\n",
    "nltk.download('punkt')"
   ]
  },
  {
   "cell_type": "raw",
   "id": "2c1b4e81",
   "metadata": {},
   "source": [
    "Now let’s define some intents of the chatbot. You can add more intents to make the chatbot more helpful and more functional:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d007afc7",
   "metadata": {},
   "outputs": [],
   "source": [
    "intents = [\n",
    "    {\n",
    "        \"tag\": \"greeting\",\n",
    "        \"patterns\": [\"Hi\", \"Hello\", \"Hey\", \"How are you\", \"What's up\"],\n",
    "        \"responses\": [\"Hi there\", \"Hello\", \"Hey\", \"I'm fine, thank you\", \"Nothing much\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"goodbye\",\n",
    "        \"patterns\": [\"Bye\", \"See you later\", \"Goodbye\", \"Take care\"],\n",
    "        \"responses\": [\"Goodbye\", \"See you later\", \"Take care\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"thanks\",\n",
    "        \"patterns\": [\"Thank you\", \"Thanks\", \"Thanks a lot\", \"I appreciate it\"],\n",
    "        \"responses\": [\"You're welcome\", \"No problem\", \"Glad I could help\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"about\",\n",
    "        \"patterns\": [\"What can you do\", \"Who are you\", \"What are you\", \"What is your purpose\"],\n",
    "        \"responses\": [\"I am a chatbot\", \"My purpose is to assist you\", \"I can answer questions and provide assistance\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"help\",\n",
    "        \"patterns\": [\"Help\", \"I need help\", \"Can you help me\", \"What should I do\"],\n",
    "        \"responses\": [\"Sure, what do you need help with?\", \"I'm here to help. What's the problem?\", \"How can I assist you?\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"age\",\n",
    "        \"patterns\": [\"How old are you\", \"What's your age\"],\n",
    "        \"responses\": [\"I don't have an age. I'm a chatbot.\", \"I was just born in the digital world.\", \"Age is just a number for me.\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"weather\",\n",
    "        \"patterns\": [\"What's the weather like\", \"How's the weather today\"],\n",
    "        \"responses\": [\"I'm sorry, I cannot provide real-time weather information.\", \"You can check the weather on a weather app or website.\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"budget\",\n",
    "        \"patterns\": [\"How can I make a budget\", \"What's a good budgeting strategy\", \"How do I create a budget\"],\n",
    "        \"responses\": [\"To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.\", \"A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.\", \"To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses.\"]\n",
    "    },\n",
    "    {\n",
    "        \"tag\": \"credit_score\",\n",
    "        \"patterns\": [\"What is a credit score\", \"How do I check my credit score\", \"How can I improve my credit score\"],\n",
    "        \"responses\": [\"A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.\", \"You can check your credit score for free on several websites such as Credit Karma and Credit Sesame.\"]\n",
    "    }\n",
    "]"
   ]
  },
  {
   "cell_type": "raw",
   "id": "012bf411",
   "metadata": {},
   "source": [
    "Now let’s prepare the intents and train a Machine Learning model for the chatbot:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "96d66800",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression(max_iter=10000, random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression(max_iter=10000, random_state=0)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression(max_iter=10000, random_state=0)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create the vectorizer and classifier\n",
    "vectorizer = TfidfVectorizer()\n",
    "clf = LogisticRegression(random_state=0, max_iter=10000)\n",
    "\n",
    "# Preprocess the data\n",
    "tags = []\n",
    "patterns = []\n",
    "for intent in intents:\n",
    "    for pattern in intent['patterns']:\n",
    "        tags.append(intent['tag'])\n",
    "        patterns.append(pattern)\n",
    "\n",
    "# training the model\n",
    "x = vectorizer.fit_transform(patterns)\n",
    "y = tags\n",
    "clf.fit(x, y)"
   ]
  },
  {
   "cell_type": "raw",
   "id": "0639b5f7",
   "metadata": {},
   "source": [
    "Now let’s write a Python function to chat with the chatbot:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "563685b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "def chatbot(input_text):\n",
    "    input_text = vectorizer.transform([input_text])\n",
    "    tag = clf.predict(input_text)[0]\n",
    "    for intent in intents:\n",
    "        if intent['tag'] == tag:\n",
    "            response = random.choice(intent['responses'])\n",
    "            return response"
   ]
  },
  {
   "cell_type": "raw",
   "id": "471bc8e5",
   "metadata": {},
   "source": [
    "Till now, we have created the chatbot. After running the code, you can interact with the chatbot in the terminal itself. To turn this chatbot into an end-to-end chatbot, we need to deploy it to interact with the chatbot using a user interface. To deploy the chatbot, I will use the streamlit library in Python, which provides amazing features to create a user interface for a Machine Learning application in just a few lines of code."
   ]
  },
  {
   "cell_type": "raw",
   "id": "196a1376",
   "metadata": {},
   "source": [
    "So, here’s how we can deploy the chatbot using Python:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1b0080bf",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-03-31 17:08:14.216 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Sanket\\anaconda3\\envs\\nlp\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "counter = 0\n",
    "\n",
    "def main():\n",
    "    global counter\n",
    "    st.title(\"Chatbot\")\n",
    "    st.write(\"Welcome to the chatbot. Please type a message and press Enter to start the conversation.\")\n",
    "\n",
    "    counter += 1\n",
    "    user_input = st.text_input(\"You:\", key=f\"user_input_{counter}\")\n",
    "\n",
    "    if user_input:\n",
    "        response = chatbot(user_input)\n",
    "        st.text_area(\"Chatbot:\", value=response, height=100, max_chars=None, key=f\"chatbot_response_{counter}\")\n",
    "\n",
    "        if response.lower() in ['goodbye', 'bye']:\n",
    "            st.write(\"Thank you for chatting with me. Have a great day!\")\n",
    "            st.stop()\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7124a3ce",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
