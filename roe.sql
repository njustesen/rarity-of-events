CREATE DATABASE roe;

USE roe;


CREATE TABLE Archive (
    EliteID VARCHAR(36) CHARACTER SET utf8 NOT NULL PRIMARY KEY,
    ExperimentID int,
    ActorID int,
    Frame int,
    EpisodeLength int,
    Timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Fitness float,
    Event0 float DEFAULT 0.0,
  	Event1 float DEFAULT 0.0,
    Event2 float DEFAULT 0.0,
    Event3 float DEFAULT 0.0,
    Event4 float DEFAULT 0.0,
    Event5 float DEFAULT 0.0,
    Event6 float DEFAULT 0.0,
    Event7 float DEFAULT 0.0,
    Event8 float DEFAULT 0.0,
    Event9 float DEFAULT 0.0,
    Event10 float DEFAULT 0.0,
    Event11 float DEFAULT 0.0
);

CREATE TABLE History SELECT * FROM Archive LIMIT 0;

CREATE USER roe@localhost IDENTIFIED BY 'RarityOfEvents';
GRANT ALL PRIVILEGES ON *.* TO roe@localhost WITH GRANT OPTION;