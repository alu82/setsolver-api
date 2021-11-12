FROM python:3.9.1

ENV FLASK_APP setsolverapi
EXPOSE 5000

COPY requirements.txt /api/requirements.txt

WORKDIR /api
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

COPY . /api
RUN pip install .

CMD ["flask", "run", "--host=0.0.0.0"]