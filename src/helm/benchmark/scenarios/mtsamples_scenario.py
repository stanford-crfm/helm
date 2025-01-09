import csv
import os
from typing import Dict, List

from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Input,
    Scenario,
    Instance,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Reference,
    Output,
)

def scrape_MTS_website(output_file):
    import time
    import pandas as pd
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    driver = webdriver.Chrome()

    def wait_for_element(xpath, timeout=10):
        """
        Wait for an element to appear on the page
        """
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, xpath)))

    def get_sample_links_from_type_page(type_url):
        """
        Given a type URL, extract all sample links from that page.
        """
        driver.get(type_url)
        wait_for_element("//table[@id='tblSamples']")  # Wait for the table to load

        sample_links = []
        sample_elements = driver.find_elements(By.XPATH, "//table[@id='tblSamples']//tbody//tr//a")
        
        for sample in sample_elements:
            sample_name = sample.text.split(' - ')[0]  # Extract sample name
            sample_link = sample.get_attribute('href')  # Extract the sample URL
            sample_links.append((sample_name, sample_link))
        
        return sample_links

    def get_sample_report(sample_link):
        """
        Given a sample link, navigate to the sample page and extract the report.
        """
        driver.get(sample_link)
        wait_for_element("//div[@class='hilightBold']")  # Wait for the report to load

        try:
            report_title = driver.find_element(By.XPATH, "//h1[contains(text(), 'Sample Name')]").text
        except:
            report_title = "Title not found"

        try:
            report_content = driver.find_element(By.XPATH, "//div[@class='hilightBold']").text
        except:
            report_content = "Content not found"

        return report_title, report_content

    def get_all_types():
        """
        Open the main page and extract all the types and their URLs.
        """
        driver.get('https://mtsamples.com/')  # Main page URL
        wait_for_element("//div[@class='col-md-6']")  # Wait for the column containing the types

        type_links = {}
        type_elements = driver.find_elements(By.XPATH, "//div[@class='col-md-6']//ol//li//a")
        
        for type_elem in type_elements:
            type_name = type_elem.text.split(' - ')[0].strip()  # Extract type name (before the dash)
            type_url = type_elem.get_attribute('href')  # Extract the type URL
            type_links[type_name] = type_url
        
        return type_links

    def save_samples_to_csv(samples):
        """
        Save the samples data to a CSV file.
        """
        df = pd.DataFrame(samples)
        df.to_csv(output_file, index=False)
        print(f"Saved {len(samples)} samples to CSV.")


    all_samples = []

    # Step 1: Get all types
    type_links = get_all_types()
    
    # Step 2: For each type, extract the sample links and their reports
    for type_name, type_url in type_links.items():
        print(f"Processing type: {type_name}")

        try:
            sample_links = get_sample_links_from_type_page(type_url)
        except Exception as e:
                print(f"Error getting sample link: {type_url}")
                print(e)
                continue 

        # Step 3: For each sample, fetch its report
        for sample_name, sample_link in sample_links:
            print(f"  Processing sample: {sample_name}")
            
            try:
                report_title, report_content = get_sample_report(sample_link)
            except Exception as e:
                print(f"    Error processing sample: {sample_name}")
                print(e)
                continue

            # Save the data in a structured way
            sample_data = {
                'type': type_name,
                'sample_name': sample_name,
                'report_title': report_title,
                'report_content': report_content
            }
            all_samples.append(sample_data)

            # Save immediately after processing each sample to avoid data loss
            save_samples_to_csv(all_samples)

    # Final save to ensure all data is written
    save_samples_to_csv(all_samples)
    driver.quit()

class MTSamplesScenario(Scenario):
    """
    MTSamples.com is designed to give you access to a big collection of transcribed medical reports. 
    These samples can be used by learning, as well as working medical transcriptionists for their daily 
    transcription needs. We present the model with patient information and request it to generate a corresponding
    treatment plan.

    Sample Synthetic Prompt:
        Given various information about a patient, return a reasonable treatment plan for the patient.
        
    dataset link: https://mtsamples.com/
    }
    """

    name = "mtsamples"
    description = "A patient care plan generation benchmark."
    tags = ["knowledge", "reasoning", "biomedical"]


    def create_benchmark(self, csv_path)->Dict[str, str]:
        data = {}
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                report_content = row['report_content']
                if "PLAN:" in report_content:
                    finding, plan = report_content.split("PLAN:", 1)
                    plan=plan.split("See More Samples on",1)[0]
                    data[finding.strip()] = plan.strip()
        return data
    
    def get_instances(self, output_path: str) -> List[Instance]:
        data_path = "/share/pi/nigam/data/mtsamples/mtsamples_with_reports_2.csv"
        if not os.path.exists(data_path):
            scrape_MTS_website(data_path)
        # ensure_directory_exists(data_path)

        instances: List[Instance] = []
        benchmark_data = self.create_benchmark(data_path)
        
        for information, plan in benchmark_data.items():
            instances.append(
                Instance(
                    input=Input(text=information),
                    references=[Reference(Output(text=plan), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                )
            )
                
        return instances
