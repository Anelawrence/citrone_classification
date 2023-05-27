import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# loading the scaler object
scaler = pickle.load(open('scaler_object.sav', 'rb')) # C:/Users/LAWRENCE/Desktop/SGA_1.3/PROJECTS_Folder/


def predict_eligibility(input_values):
    input_value_as_array = np.asarray(input_values)
    input_data_reshape = input_value_as_array.reshape(1,-1)
    # Standardizing the user input with standard scaler
    input_data_reshape = scaler.transform(input_data_reshape)
    predict_value = loaded_model.predict(input_data_reshape)
    print(predict_value)
    if predict_value[0]==0:
       return 'Student not eligible'
    else:
       return 'Student eligible for intermediate'
    

def main():
   
    # giving a title
    st.title('Intermediate Class Eligibility Prediction')
    st.text_input('Enter your Name:', key='name')
    

    #Getting inpit data from user
    Lesson_Summary=st.number_input('Student lesson summary(0-10)')
    Assignment_Summary=st.number_input('Student Assignment Summary')
    Grade_Point_Average=st.number_input('Student Average Grade Point')

    #code for prediction
    eligibility=''

    # create button for prediction
    if st.button('Check Eligibility'):
        eligibility = predict_eligibility([Lesson_Summary, Assignment_Summary, Grade_Point_Average])

    st.success(eligibility)



if __name__== '__main__':
    main()


st.write(f"Have a nice day {st.session_state.name}")
    