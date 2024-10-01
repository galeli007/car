import streamlit as st
import requests
import json
import openai
import asyncio
import nest_asyncio
import nltk
import logging
import os
import pandas as pd
import time
from tqdm.asyncio import tqdm_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, filename='clinical_trials.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set your OpenAI API key securely
openai.api_key = os.getenv('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')
if not openai.api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

def clean_text(text):
    """
    Cleans and formats text by removing escape characters and unnecessary whitespace.
    """
    if isinstance(text, str):
        return text.replace('\\>', '>').replace('\\<', '<').replace('\\', '').strip()
    return text

# Function to generate medical condition keywords from patient data using OpenAI's GPT
def generate_keywords(patient_data):
    """
    Generate medical condition keywords from patient data using OpenAI's API.
    """
    prompt = (
        f"Extract medical condition keywords from the following patient data:\n\n"
        f"{patient_data}\n\n"
        f"Medical Conditions (comma-separated):"
    )
    try:
        response = openai.chat.completions.create(
            model='gpt-3.5-turbo',  # Replace with 'gpt-4' if available
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.5
        )
        keywords_text = response.choices[0].message.content.strip()
        keywords_list = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
        logging.info(f"Generated Keywords: {keywords_list}")
        return keywords_list
    except Exception as e:
        logging.error(f"Error generating keywords: {e}")
        print(f"Error generating keywords: {e}")
        return []

# Function to read NCT Numbers from CSV
def load_nct_numbers(csv_file_path):
    """
    Load NCT Numbers from a CSV file.

    Parameters:
    - csv_file_path (str): Path to the CSV file.

    Returns:
    - nct_numbers (list): List of NCT Numbers.
    """
    df = pd.read_csv(csv_file_path)
    nct_numbers = df['NCT Number'].dropna().unique().tolist()
    logging.info(f"Loaded {len(nct_numbers)} NCT Numbers from CSV.")
    return nct_numbers


def fetch_trials_by_nct_numbers(nct_numbers,recruitment_status='RECRUITING'):
    """
    Fetch trial details for given NCT Numbers.

    Parameters:
    - nct_numbers (list): List of NCT Numbers.

    Returns:
    - trials_data (list): List of trial details.
    """
    base_url = 'https://clinicaltrials.gov/api/v2/studies'
    trials_data = []
    batch_size = 50  # Adjust as needed

    # Load cached trials if available
    cache_file = 'trials_cache.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_trials = json.load(f)
        cached_nct_ids = set(trial['protocolSection']['identificationModule']['nctId'] for trial in cached_trials)
    else:
        cached_trials = []
        cached_nct_ids = set()

    # NCT IDs that need to be fetched
    nct_numbers_to_fetch = [nct for nct in nct_numbers if nct not in cached_nct_ids]

    for i in range(0, len(nct_numbers_to_fetch), batch_size):
        batch_nct_ids = nct_numbers_to_fetch[i:i+batch_size]
        params = {
            'format': 'json',
            'filter.ids': ','.join(batch_nct_ids),
            'filter.overallStatus': recruitment_status,
            'fields': (
                'protocolSection.identificationModule.nctId,'
                'protocolSection.identificationModule.briefTitle,'
                'protocolSection.identificationModule.officialTitle,'
                'protocolSection.conditionsModule.conditions,'
                'protocolSection.descriptionModule.briefSummary,'
                'protocolSection.eligibilityModule.eligibilityCriteria'
                
            )
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            logging.error(f"Error fetching data: HTTP {response.status_code}")
            print(f"Error fetching data: HTTP {response.status_code}")
            print(f"Response content: {response.text}")
            continue
        data = response.json()
        studies = data.get('studies', [])
        trials_data.extend(studies)
        time.sleep(0.2)  # Be polite and avoid hitting the API too hard

    # Combine cached and newly fetched trials
    all_trials = cached_trials + trials_data

    # Save the combined trials to cache
    with open(cache_file, 'w') as f:
        json.dump(all_trials, f)

    logging.info(f"Fetched {len(trials_data)} new trials. Total trials: {len(all_trials)}")
    return all_trials



# Function to filter trials based on condition keywords
def filter_trials_by_conditions(trials, keywords):
    """
    Filter trials based on condition keywords.

    Parameters:
    - trials (list): List of trial details.
    - keywords (list): List of condition keywords.

    Returns:
    - filtered_trials (list): List of trials matching the keywords.
    """
    filtered_trials = []
    keyword_set = set(kw.lower() for kw in keywords)
    for trial in trials:
        conditions = trial.get('protocolSection', {}).get('conditionsModule', {}).get('conditions', [])
        condition_set = set(cond.lower() for cond in conditions)
        if condition_set & keyword_set:
            filtered_trials.append(trial)
    logging.info(f"Filtered {len(filtered_trials)} trials based on condition keywords.")
    return filtered_trials

# Function to extract relevant information from each study
def extract_trial_info(study):
    """
    Extract relevant information from a study.
    """
    protocol_section = study.get('protocolSection', {})
    identification_module = protocol_section.get('identificationModule', {})
    description_module = protocol_section.get('descriptionModule', {})
    conditions_module = protocol_section.get('conditionsModule', {})
    eligibility_module = protocol_section.get('eligibilityModule', {})

    nct_id = identification_module.get('nctId', 'N/A')
    brief_title = identification_module.get('briefTitle', 'N/A')
    official_title = identification_module.get('officialTitle', 'N/A')
    conditions = conditions_module.get('conditions', [])
    brief_summary = description_module.get('briefSummary', 'N/A')
    eligibility_criteria = eligibility_module.get('eligibilityCriteria', 'N/A')

    # Clean and format text fields
    brief_summary = clean_text(brief_summary)
    eligibility_criteria = clean_text(eligibility_criteria)

    # Concatenate conditions if multiple
    conditions_str = '; '.join(conditions) if isinstance(conditions, list) else conditions

    trial_info = {
        'nct_id': nct_id,
        'title': official_title or brief_title,
        'conditions': conditions_str,
        'brief_summary': brief_summary,
        'eligibility_criteria': eligibility_criteria
    }
    return trial_info

# Asynchronous function to evaluate patient eligibility for a trial using OpenAI's GPT
async def match_patient_to_trial_async(patient_data, eligibility_criteria):
    """
    Determine if a patient matches a trial's eligibility criteria using OpenAI's API.
    """
    prompt = (
        f"Determine if the patient is eligible for the clinical trial based on the eligibility criteria.\n\n"
        f"Patient Information:\n{patient_data}\n\n"
        f"Eligibility Criteria:\n{eligibility_criteria}\n\n"
        f"Analyze each criterion individually and conclude with 'Eligible' or 'Not Eligible' along with a brief explanation referencing the specific criteria."
    )

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai.api_key)
        response = await client.chat.completions.create(
            model='gpt-3.5-turbo',  # Use 'gpt-4' if you have access
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        print(f"Error matching patient to trial: {e}")
        return "Not Eligible: Unable to determine eligibility due to an error."

# Asynchronous function to rank trials based on patient eligibility
async def rank_trials(patient_data, trials):
    """
    Rank trials based on patient eligibility.
    """
    trial_infos = []
    for trial in trials:
        trial_info = extract_trial_info(trial)
        eligibility_criteria = trial_info['eligibility_criteria']
        if not eligibility_criteria or eligibility_criteria == 'N/A':
            continue
        trial_infos.append((trial_info, eligibility_criteria))

    logging.info(f"Matching patient to {len(trial_infos)} trials.")
    print("Matching trials...")
    trial_results = []

    semaphore = asyncio.Semaphore(5)  # Limit concurrent API calls to prevent rate limiting

    async def process_trial(trial_info, eligibility_criteria):
        async with semaphore:
            match_result = await match_patient_to_trial_async(patient_data, eligibility_criteria)
            is_eligible = 'Eligible' in match_result
            score = 1 if is_eligible else 0
            trial_data = {
                'nct_id': trial_info['nct_id'],
                'title': trial_info['title'],
                'score': score,
                'match_result': match_result
            }
            return trial_data

    tasks = [process_trial(info, criteria) for info, criteria in trial_infos]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        trial_data = await f
        trial_results.append(trial_data)

    # Sort trials by eligibility score in descending order
    trial_results.sort(key=lambda x: x['score'], reverse=True)
    logging.info("Completed matching trials.")
    return trial_results
def simplify_summary_in_layman_terms(trial_summary):
    if not trial_summary:
        return "No summary available."

    prompt = (
        f"Simplify the following clinical trial description into layman's terms:\n\n"
        f"{trial_summary}\n\n"
        f"Simplified Description:"
    )
    try:
        response = openai.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        simplified_summary = response.choices[0].message.content.strip()
        logging.info(f"Simplified Summary: {simplified_summary}")
        return simplified_summary
    except Exception as e:
        logging.error(f"Error simplifying summary: {e}")
        return "Unable to simplify the summary due to an error."
# Main application
def main():
    st.title("Clinical Trials Matching App")

    # Input for patient data
    patient_data = st.text_area("Enter patient data here:", height=200)

    if st.button("Submit"):
        if not patient_data.strip():
            st.warning("Please enter patient data.")
            return

        st.info("Generating keywords from patient data...")
        keywords = generate_keywords(patient_data)
        if not keywords:
            st.error("Failed to generate keywords.")
            return
        st.success(f"Keywords generated: {', '.join(keywords)}")

        # Load NCT Numbers and trial data
        csv_file_path = 'table_trials_andy.csv'  # Replace with your actual CSV file path
        nct_numbers = load_nct_numbers(csv_file_path)
        trials_data = fetch_trials_by_nct_numbers(nct_numbers)

        # Filter trials based on condition keywords
        st.info("Filtering trials based on condition keywords...")
        filtered_trials = filter_trials_by_conditions(trials_data, keywords)
        st.success(f"Filtered {len(filtered_trials)} trials based on condition keywords.")

        if not filtered_trials:
            st.warning("No trials found matching the condition keywords.")
            return

        # Limit the number of trials to process to manage API usage
        max_trials_to_process = 10  # Adjust as needed
        trials_to_process = filtered_trials[:max_trials_to_process]

        st.info(f"Matching patient to {len(trials_to_process)} trials...")

        # Run the ranking asynchronously
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ranked_trials = loop.run_until_complete(rank_trials(patient_data, trials_to_process))
        except Exception as e:
            st.error(f"An error occurred during trial matching: {e}")
            return

        if not ranked_trials:
            st.warning("No matching trials found.")
            return

        st.success("Top Matching Clinical Trials (with simplified descriptions):")

        for idx, trial in enumerate(ranked_trials, start=1):
            st.subheader(f"{idx}. {trial['title']}")
            st.write(f"**NCT ID:** {trial['nct_id']}")
            eligibility = 'Eligible' if trial['score'] == 1 else 'Not Eligible'
            st.write(f"**Eligibility:** {eligibility}")
            st.write(f"**Match Result:**\n{trial['match_result']}")

            # Simplify the trial's brief summary
            simplified_summary = simplify_summary_in_layman_terms(trial.get('brief_summary', ''))
            st.write(f"**Simplified Trial Description:**\n{simplified_summary}\n")

if __name__ == '__main__':
    main()