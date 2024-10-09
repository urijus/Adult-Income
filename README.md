# Adult-Income
Predict whether income exceeds $50K/yr based on census data

## Description

add description

## Getting Started

This section will guide you through setting up the project on your local machine.

### Prerequisites

- Python 3.10
- pip (Python package installer)

### Installation

1. Clone the Ignion repository to your local machine:
    ```bash
    git clone https://git.basetis.com/ai-team/ignion-ai-assistant.git
    ```

2. Navigate to the project directory:
    ```
    cd ignion-ai-assistant
    ```

3. Create a virtual environment:
    ```bash
    python3 -m venv venv
    ```

4. Activate the virtual environment:
   - Windows:
       ```bash
       venv\Scripts\activate
       ```
   - Linux/macOS:
      ```bash
      source venv/bin/activate
      ```

5. Install the required dependencies:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## Project Setup

This section covers the initial setup steps necessary after installing the project, 
including creating required directories and setting up environment variables.

### Configuration Setup

You also need to set up the `config.yaml` file to specify the data prefix where the
necessary documents and files are stored.

1. Create a `config.yaml` file in the `src` directory:

    ```bash
    touch src/config.yaml
    ```

2. Add the following content to the config.yaml file:

    ```bash
    data_prefix: ""
    ```

3. Set the data_prefix variable to the appropriate path based on your environment (local or remote).

### Creating Required Directories

First, you need two main directories: `notebooks` and the `data` directory. The `data` directory 
must coincide with the `data_prefix` specified in the `config.yaml` file.

- **Data Directory**: This directory stores documents for the database, the vector database 
after its creation, the stored users conversation states, the credentials to access the GUI,
the examples of user projects and other documents. It includes four subdirectories: `access_keys`, 
`documents`, `user_projects`, `user_states` and `vector_db`.

    ```bash
    cd data_prefix # Replace ~data_prefix~ with the path specified in the config.yaml file
    mkdir -p access_keys documents user_projects user_states vector_db
    ```
  
Inside the `access_keys` directory, you need to store the credentials to access the GUI. The credentials 
are in json format and have the following structure:

```bash
touch access_keys/assistant_credentials.json
```

The file stores the username and the hash password of the users. The file has the following structure:

```json
  {
      "username1": "hash_password1",
      "username2": "hash_password2"
  }
```

If the code is running in a local environment, in order to store the conversation states, you need to create
a json file called `state_examples.json` in the `user_states` directory. This file will store the conversation
states of the users. The file has the following structure:

```bash
touch user_states/state_examples.json
```

- **Notebooks Directory**: This directory is intended for storing notebooks with code 
tests and experiments.

    ```bash
    mkdir notebooks
    ```

### Environment Setup

After setting up the directories, you need to configure the environment variables:

1. Create a `.env` file in the project root directory:

    ```bash
    touch .env
    ```

2. Copy the structure from `.env_template` to the `.env` file:

    ```bash
    cp .env_template .env
    ```

3. Fill in the `.env` file with the required API keys. Ensure you have the necessary 
4. API keys and permissions before proceeding.

Follow these steps to ensure your project is ready for development and experimentation.

## Usage

This section will provide examples of how to use the project once it is set up.

### Running the AI assistant through the command line

To run the AI assistant through the command line, there is a script available in the 
`scripts` directory called `run_assistant`. You can run the script using the following command:

```bash
python scripts/run_assistant.py
```

This script will start the AI assistant in the command line interface, allowing you to
interact with it using text input.

### Running the AI assistant through the web interface

To run the AI assistant through the web interface, you can start the API server using the
following command:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 13101 --reload
```

This command will start the API server on `http://127.0.0.1:13101`, allowing you to interact
with the AI assistant through HTTP requests.

To run the GUI interface, you can start the web server using the following command:

```bash
streamlit run api/streamlit_app.py --server.port 13201
```

This command will start the Streamlit web server, allowing you to interact with the AI assistant
through a graphical user interface.
