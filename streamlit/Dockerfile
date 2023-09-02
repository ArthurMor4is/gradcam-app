FROM python:3.9.5

ENV PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt requirements.txt

RUN /usr/local/bin/python -m pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8501

COPY . .

CMD ["streamlit", "run", "grad_cam.py"]
