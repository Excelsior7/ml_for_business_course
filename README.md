# ML for Business II Project 

## Business Use Case : Salary prediction based on job description

In today's competitive job market, one of the most pressing concerns for job seekers is determining a realistic salary expectation for a given position. This information is crucial, not only to ensure that job seekers can negotiate effectively but also to set career expectations and make informed decisions about job applications. For employers, understanding salary trends can help in setting competitive offers that attract top talent.

This project seeks to address the following questions:

- What is a reasonable annual salary for a particular job posting?

- Which factors have the greatest influence on salary levels?

- How can job seekers get a more transparent view of the salary landscape across industries and positions?

### Objective:

The primary goal of this project is to predict the expected salary for a given job posting based on a variety of features, including job title, company and location. By creating a machine learning model that leverages this information, we aim to provide users with a predicted salary, which will aid in making better career decisions.

### Potential Impact:

- Job Seekers: Gain insights into salary expectations, helping them better negotiate their salaries and focus on job opportunities that align with their financial goals.

- Employers/HR Teams: By understanding the salary landscape, companies can offer competitive packages to attract the right talent.

- Recruitment Agencies: Obtain data-driven insights into salary expectations across industries, which will help in providing better guidance to clients and candidates.

### Why It Matters:

In many job postings, salary information is often omitted or vague, leading to job seekers having unrealistic expectations or feeling undercompensated once an offer is made. With a data-driven model, this project aims to bring transparency to salary expectations, thus reducing mismatches between employer offers and candidate expectations.

## Description of the dataset used : LinkedIn Job Postings (2023 - 2024)

The dataset used for this project is the LinkedIn Job Postings Dataset, sourced from Kaggle. This dataset provides comprehensive information about job listings from LinkedIn, including key details about the job positions, company characteristics, and salary-related information. The dataset contains various attributes that can be used to predict expected salaries for job positions.

### Dataset Characteristics

- Rows (Job Postings): 123,849 job postings across various industries, locations, and job types.
- Columns (Features): 31 features, providing a wide variety of information, from structured data like salary and location to unstructured data like job descriptions.

### Types of Features:

#### Numerical Features:

- Salary information (min_salary, max_salary, med_salary, normalized_salary):
- Job engagement metrics (views, applies)
- Geographical identifiers (zip_code)

#### Categorical Features:
- Job details (title, company_name, location, formatted_work_type)
- Payment frequency (pay_period, currency)
- Work-related features (remote_allowed, work_type)

#### Text Features:
- Job description (description): A long-form text that explains the responsibilities, qualifications, and skills needed for the job.
- Skills description (skills_desc): A list of required skills for the position, if provided.

### Data we kept: 

- company_name: Company name
- title: Job title
- description: Job description
- pay_period: Pay period for salary (Hourly, Monthly, Yearly)
- max_salary: Maximum salary
- med_salary: Median salary
- min_salary: Minimum salary
- location: Job location
- remote_allowed: Whether job permits remote work
- work_type: Type of work associated with the job (Fulltime, Parttime, Contract)
- currency: Currency in which the salary is provided

### Exploratory Data Analysis choices

- In every job posting including salary informations we either had a salary range (min_salary and max_salary) or a med_salary. In both cases we used that info to calculate a **normalized_salary** wich is a year salary so it takes into account the pay_period  (Hourly, Monthly, Yearly). We decided then to predict the **normalized_salary** and to get rid of pay_period, max_salary, med_salary and min_salary

- We only kept postings with non null salary information: 36073 / 123849

- We decided to only keep USD currency information: 36058 / 36073

- We focused on full time, contract and part time job posting (FULL_TIME: 29119, CONTRACT: 3848 and PART_TIME: 2304)

- We kept standard salaries that were between the quantile(0.05) and quantile(0.99).

- We used getdummies to encode work_type

- From location we decided to keep only state information and to concatenate that information with job title, company name and description. 

### Target

We want to predict the normalized_salary. 