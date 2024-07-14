import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.vectorstores import faiss
from langchain.chains  import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import huggingface
from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd
import cv2 as cv
from ultralytics import YOLO
import supervision as sv
model = GoogleGenerativeAI(google_api_key='AIzaSyDQViky6jXUHMHNSlD_tcTUQ06QYa2Ya_A',temperature=0,model='models/text-bison-001')

embedder = huggingface.HuggingFaceBgeEmbeddings()
filepath = 'faissindex'


#functions  
def AddDish():
    st.header("Cant Find the DISH ???")
    st.subheader("Add it right now with the click of a button")
    input1 = st.text_input("Enter the name of the dish")
    input2 = st.text_input("Enter the ingredients required")
    input3 = st.text_input("Vegetarian or Non-Vegetarian??")
    input4 = st.text_input("Preparation Time (A guess would also be ok)")
    input5 = st.text_input("Cooking time (A guess would be ok )")
    input6 = st.text_input("What is it Sweet/Sour/Spicy ???")
    input7 = st.text_input("MainCOurse/desert ???")
    input8 = st.text_input("what state does this dish come from")
    input9 = st.text_input("North , South ,East or West ????")
    df = pd.read_csv("indian_food.csv")
    button1 = st.button(label="Submit")
    new_row = pd.DataFrame( {
            'name': [input1],
            'ingredients': [input2],
            'diet': [input3],
            'prep_time': [input4],
            'cook_time': [input5],
            'flavor_profile': [input6],
            'course': [input7],
            'state': [input8],
            'region': [input9]
        })
    if button1:
        df = pd.concat([new_row])
        st.success("Submitted")






def get_qa():
    vector_db = faiss.FAISS.load_local(filepath,embeddings=embedder,allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever()
    prompt = '''Given the following dish generate the recipe and the ingredients all of them using headings and 
    bullet points , Give the steps that can be followed to achieve the tasty dish use all the information from the given 
    source document.Give a youtube link in the end so that it can be followed. The youtube link must be at the top of the recomandation line
    of youtube
    Context : {context} 
    Question :{question}
    '''
    template = PromptTemplate(
        template=prompt,
        input_variables=['context','question']      
    )

    chain = RetrievalQA.from_chain_type(
        llm=model,chain_type ="stuff",
        retriever=retriever,input_key="query",
        return_source_documents=False,
        chain_type_kwargs = {"prompt":template})

    return chain
def home():
    st.header("Dish UP")
    entered = st.text_input("Enter the name of dish")
    clicked = st.button("Find")
    if clicked:
        chain = get_qa()
        result = chain(entered)
        sentence = ''
        for word in result['result']:
            if word != '\n':
                sentence += word
            else:
                st.subheader(sentence)
                sentence = ''

def objectDetection():
    model = YOLO(model='C:/Users/sarva/OneDrive/Desktop/Codes/Intel Unnati Project/best.pt')
    
    bouding_box = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    
    frame_placeholder = st.empty()
    video = cv.VideoCapture(0)
    while video.isOpened():
        ret,frame = video.read()

        if ret == True:
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)

            frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            annotated_image = bouding_box.annotate(scene=frame,detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image,detections=detections)

            frame_placeholder.image(annotated_image,channels='RGB')

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    video.release()
    cv.destroyAllWindows


menu = ['Home','Add dish','Detect']
choice = st.sidebar.selectbox("Explore More",menu)
if choice == 'Add dish':
    AddDish()
elif choice == 'Home':
    home()  
elif choice == 'Detect':
    objectDetection()