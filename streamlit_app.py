import streamlit as st
import requests
import json

# Get unique state codes from the mapping
def get_unique_state_codes(mapping_dict):
    return sorted(set(mapping_dict.values()))

def load_location_mapping():
    with open('location_renaming_mapping.json', 'r') as file:
        return json.load(file)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Salary Prediction Tool",
        page_icon="💰",
        layout="wide"
    )
    
    # Load location mapping and get unique state codes
    try:
        location_mapping = load_location_mapping()
        state_codes = get_unique_state_codes(location_mapping)
    except FileNotFoundError:
        st.error("Location mapping file not found. Using default state codes.")
        state_codes = ["CA", "NY", "TX", "FL", "IL", "WA", "MA", "CO", "GA", "NC"]
    
    # Header section
    st.title("💰 Salary Prediction Tool")
    
    # Information expander
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        ### Predict Job Salaries with AI
        This tool helps predict salary ranges for job positions based on key factors:
        * Job title and description
        * Company details
        * Location (US state codes)
        * Work arrangement (remote/on-site)
        * Employment type
        
        Fill in the form below to get a predicted salary range for your job posting.
        """)
    
    # Main form in a clean container
    with st.container():
        st.markdown("### 📝 Enter Job Details")
        
        with st.form("job_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Position Information")
                title = st.text_input(
                    "Job Title",
                    value="Data Analyst",
                    help="Enter the specific job title (e.g., Data Analyst, Software Engineer)"
                )
                
                company_name = st.text_input(
                    "Company Name",
                    value="TechCorp",
                    help="Enter the company name"
                )
                
                # Simple state code selector
                state = st.selectbox(
                    "State Code",
                    options=state_codes,
                    help="Select the state code (e.g., CA for California)",
                    index=state_codes.index("CA") if "CA" in state_codes else 0
                )
            
            with col2:
                st.markdown("#### 💼 Work Arrangements")
                remote_allowed = st.radio(
                    "Remote Work Option",
                    options=["Yes", "No"],
                    horizontal=True,
                    index=0,
                    help="Indicate if remote work is allowed"
                )
                
                work_type = st.radio(
                    "Employment Type",
                    options=["Full Time", "Part Time", "Contract"],
                    horizontal=True,
                    index=0,
                    help="Select the type of employment"
                )
            
            st.markdown("#### 📋 Job Description")
            description = st.text_area(
                "Detailed Job Description",
                value="Analyze data and create reports",
                height=120,
                help="Provide a detailed description of the role, responsibilities, and requirements"
            )
            
            # Submit button with styling
            submitted = st.form_submit_button(
                "🔍 Predict Salary",
                use_container_width=True
            )
            
            if submitted:
                # Prepare data to send to the API
                data = {
                    "remote_allowed": True if remote_allowed == "Yes" else False,
                    "work_type_contract": work_type == "Contract",
                    "work_type_full_time": work_type == "Full Time",
                    "work_type_part_time": work_type == "Part Time",
                    "state": state,
                    "company_name": company_name,
                    "title": title,
                    "description": description
                }
                
                # API URL
                url = "https://ml-for-business-course.onrender.com/predict"
                
                # Send the request to the API
                try:
                    response = requests.post(url, json=data)
                    
                    if response.status_code == 200:
                        # API returns a float representing the predicted salary
                        predicted_salary = response.json()
                        
                        # Display results in an attractive format
                        st.success("✅ Salary Prediction Complete!")
                        
                        # Format and display the predicted salary
                        st.metric(
                            "Predicted Salary",
                            f"${round(predicted_salary["salary"],2)}"  # Formats the float as a currency
                        )
                        
                        # Show the raw data in an expander
                        with st.expander("🔍 View Raw Data"):
                            st.json(data)
                    
                    else:
                        st.error(f"API Error: {response.status_code}")
                
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")

if __name__ == "__main__":
    main()
