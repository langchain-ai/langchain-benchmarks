from langsmith import Client

if __name__ == "__main__":
    client = Client()
    dataset = client.upload_csv(
        csv_file="data.csv",
        input_keys=["input_question"],
        output_keys=["output_text"],
        name="Titanic CSV Data",
        description="QA over titanic data",
        data_type="kv",
    )
