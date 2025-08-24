# Dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.13
# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container at /app
COPY requirements.txt .

# 4. Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application's code into the container
# This includes main.py, the scripts/ and data/ folders, and mlruns/
COPY . .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Define the command to run your app
# This command runs the Uvicorn server and makes it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]