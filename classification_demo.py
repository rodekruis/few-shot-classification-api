import streamlit as st
import requests


def build_prompt(class_names, class_examples, input_text):
    model_prompt = ""
    for class_name, example in zip(class_names, class_examples):
        model_prompt += f"Message: {example}\n"
        model_prompt += f"Class: {class_name}\n"
        model_prompt += f"###\n"
    model_prompt += f"Message: {input_text}\n"
    model_prompt += f"Class:"
    return model_prompt


st.header("text-classification-app")
st.markdown("Use language models to classify text.")
st.markdown("Built with love by [NLRC 510](https://www.510.global/). See [the project on GitHub](https://github.com/rodekruis/text-generation-app).")
nclass = st.number_input('How many classes do you want to define?', min_value=0, max_value=10, value=0, step=1)
if nclass > 0:
    st.markdown("Please assign a name and a meaningful example to each class.")

    class_names = []
    class_examples = []
    col1, col2 = st.columns(2)
    for i in range(nclass):
        with col1:
            class_names.append(st.text_area('Class name', key=f"class_name_{i}"))
        with col2:
            class_examples.append(st.text_area('Example', key=f"class_example_{i}"))

    input_text = st.text_area('Input text', 'Hello world!')
    submit = st.button('Classify')
    if submit:

        url = 'http://language-model.westeurope.cloudapp.azure.com/generate/'
        payload = {
            'text': build_prompt(class_names, class_examples, input_text),
            'length': max([len(x) for x in class_names])
        }
        print(payload)

        with st.spinner(text="This may take a moment..."):
            result = requests.post(url, json=payload).json()
            if 'generated_text' in result.keys():
                text = result['generated_text']
            else:
                text = "An error occurred, please try again later or [contact us](mailto:support@510.global)!"

        st.text_area('Output', text)