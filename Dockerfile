FROM rasa/rasa:2.6.1

USER root
RUN pip install fuzzywuzzy
ENTRYPOINT bash