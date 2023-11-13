# Groceries in a blizzard - Minneapolis, MN

Minneapolis, MN receives a ton of snow every year. 
While most residents will try to remain at home during a snow storm, there will be instances where they need to go out, get in their car, and drive to a nearby grocery store to 
resupply. The goal of this project is to determine the safest route a resident can take to a nearby grocery store in a snow storm.

A full description of the project can be found in app.py or in the running demo of the app located at https://groceries-in-a-blizzard.onrender.com/. Note: the demo is run on limited cpu/ram resources, which means it is quite slow to run ML optimizations and will take a while to run (~1-3min per address). A local installation of this repo will run *significantly* faster, but will require an OpenRoute Service API token (https://openrouteservice.org/).

### Installation:
- Python version 3.10.13 or above
- pip install r- requirements.txt
- set environment variable ORS_API to an OpenRoute Service API token

### Running:
- streamlit run app.py
