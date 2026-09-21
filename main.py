from graph import app
from pprint import pprint


def main():
    print("Hello from langchain-corrective-rag!")

    result = app.invoke(input={"question": "Does Red Hat security development lifecycle methodology include DAST?"})
    pprint(result['answer'])

if __name__ == "__main__":
    main()
