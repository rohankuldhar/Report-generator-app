from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.llms import HuggingFaceHub, OpenAI
from langchain.chains.question_answering import load_qa_chain
import os
from langchain_community.vectorstores import Chroma
from googlesearch import search
import csv
import streamlit as st
import pandas as pd

def get_links(name):

        # to search
        queries = [f"Identify the industry in which {name} operates, along with its size, growth rate, trends, and key players.",
                  f"Analyze {name}'s main competitors, including their market share, products or services offered, pricing strategies, and marketing efforts",
                  f"Identify key trends in the {name}'s market, including changes in consumer behavior, technological advancements, and shifts in the competitive landscape.",
                  f"Gather information on {name}'s financial performance, including its revenue, profit margins, return on investment, and expense structure"]

        links = []

        #getting one link from each query

        for i in queries:
          for j in search(i, tld="com", num=1, stop=1, pause=1):
              links.append(j)

        st.write('Required links generated')
        for i in links:
             st.write(i)

        return links

def create_database(links):

        database={}

        for url in links:
          lst=[]
          lst.append(url)
          loader = UnstructuredURLLoader(urls=lst)
          data = loader.load()
          database[url]=data

        with open('Database1.csv', 'w', newline='',encoding="utf-8") as csvfile:
            header_key = ['URL', 'Data']
            new_val = csv.DictWriter(csvfile, fieldnames=header_key)

            new_val.writeheader()
            for new_k in database:
                new_val.writerow({'URL': new_k, 'Data': database[new_k]})

        st.write('Database created at local directory')


def text_spliter():
        raw_data=pd.read_csv('Database1.csv')
        data=raw_data['Data'].tolist()
        text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    separators=['\n\n', '\n', '.', ','],
                    )
        chunks = text_splitter.create_documents(data)
        return chunks

def main():
        st.header("Report generator app")
        name=st.text_input('Enter company name - Example. canoo,microsoft,google etc')

        if name:

          # getting links from internet
          links=get_links(name)

          # creating local database by scraping links
          data=create_database(links)
          chunks=text_spliter()

          # converting text into vector matrix
          embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
          # embeddings = OpenAIEmbeddings()

          VectorStore=Chroma.from_documents(chunks, embeddings)

          st.write('Embeddings calculated')

          os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
          #os.environ['OPENAI_API_KEY'] = 'your key'


          llm = HuggingFaceHub(repo_id="facebook/bart-large-cnn") # summarization model
          #llm = OpenAI()

          chain = load_qa_chain(llm=llm, chain_type="stuff")

          queries = [
              f"What is the industry size of {name}?",
              f"Who are the main competitors of {name}",
              f"What are the key market trends related to {name}?",
              f"How is {name}'s financial performance?"
          ]

          report={}

          for query in queries:

            # similarity search to get data relevant to query
            docs = VectorStore.similarity_search(query=query, k=10)
            response = chain.run(input_documents=docs, question=query)
            report[query]=response
            st.write(query)
            st.write(response)

          # creating report in csv format
          with open('Report.csv', 'w', newline='') as csvfile:
              header_key = ['Prompt', 'Response']
              new_val = csv.DictWriter(csvfile, fieldnames=header_key)

              new_val.writeheader()
              for new_k in report:
                  new_val.writerow({'Prompt': new_k, 'Response': report[new_k]})

          st.write('Report generated in current directory in csv format by name Report.csv')

if __name__ == '__main__':
    main()