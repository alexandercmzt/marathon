#!/usr/bin/env python
# coding: utf-8

# Author: Caitrin Armstrong, McGill ID 260501112

import csv
from collections import defaultdict
import datetime
import re
import numpy as np
import pickle
import unicodedata


def main():
    """
    Reads in participant data and calls processing method.
    :return: null
    """
    participants = dict()
    with open('/Users/Caitrin/desktop/marathon/docs/Project1_data.csv') as csvfile:
        reader = csv.reader(csvfile)
        reader.next()
        for row in reader:
            participants[row[0]] = [row[i:i + 5] for i in xrange(1, len(row), 5)]

    #False if wanting to return final 2016 data.
    isTraining = True
    createFeatureArrays(participants, isTraining)

def createFeatureArrays(participants, isTraining):
    """
    Processes data. Step-by-step documentation.
    As a side-effect will generate pickle files of numpy arrays:

    if training = true
        X_participantDataForRaceParticipation.p
        X_participantDataForRaceTimes.p
        Y_montrealMarathonParticipaton.p
        Y_montrealMarathonTime.p

    if training = false:
        final_participantDataFinalFor2016.p

    :param participants: 2d array of race participation by participant
    :param isTraining: boolean if training data is required
    :return: None
    """

    marathonTotalTime, marathonAverageTime = getAverageMarathonTimes(participants)

    # outputs for clarity's sake. 2d lists, will be converted to numpy arrays
    participantDataForRaceParticipation = list()
    participantDataForRaceTimes = list()  # without those who did not participate in 2015
    montrealMarathonParticipaton = list()
    montrealMarathonTime = list()
    participantDataFinalFor2016 = list()  # years shifted by one. So including 2015, excluding 2012.

    for participantNumber, raceList in participants.items():
        year2012, year2013, year2014, year2015, year2016 = (0,) * 5
        gender = 1  # default male
        ageSum = 0
        distanceFromAverageSum = 0  # in seconds, for this race.
        numberOfMarathons = 0
        numberOfMontrealMarathons = 0
        totalNumberOfAgedRaces = 0
        montreal2015FinishTime = 0
        attendedMontreal2015 = False

        for race in raceList:
            if "2012" in race[0]:
                year2012 += 1
            elif "2013" in race[0]:
                year2013 += 1
            elif "2014" in race[0]:
                year2014 += 1
            elif "2015" in race[0]:
                year2015 += 1
                if "Marathon Oasis Rock 'n' Roll de Montreal".lower() in strip_accents(race[1].lower()) \
                        and "marathon" in race[2].lower() \
                        and "demi" not in race[2].lower():
                    raceName = strip_accents(race[1].lower())
                    attendedMontreal2015 = True
                    montreal2015FinishTime = getFinishTime(race[3], raceName, marathonTotalTime)
                    if isTraining:  # if this is the training set, set their time to the average time.
                        year2015 -= 1
                        distanceFromAverageSum += 0
                        numberOfMarathons += 1
                        raceAge, potentialGender = processAge(race[4])
                        if raceAge is not None:
                            ageSum += raceAge
                            totalNumberOfAgedRaces += 1
                        if potentialGender is not None:
                            gender = potentialGender
                        continue
            elif "2016" in race[0]:
                year2016 += 1

            raceAge, potentialGender = processAge(race[4])
            if raceAge is not None:
                ageSum += raceAge
                totalNumberOfAgedRaces += 1
            if potentialGender is not None:
                gender = potentialGender

            if "marathon" in race[2].lower():
                numberOfMarathons += 1
                if "montreal" in strip_accents(race[1].lower()):
                    numberOfMontrealMarathons += 1
                if any(word in race[2].lower() for word in ["demi", "half"]):
                    continue                    
                else:
                    raceName = strip_accents(race[1].lower())
                finishSeconds = getFinishTime(race[3], raceName, marathonTotalTime)
                distanceFromAverageSum += finishSeconds - marathonAverageTime[raceName]

        #  Now finished processing all races for this participant. Append to final arrays.
        if numberOfMarathons != 0:
            distanceFromAverageAverage = float(distanceFromAverageSum) / float(numberOfMarathons)
        if totalNumberOfAgedRaces != 0:
            age = ageSum / totalNumberOfAgedRaces
        else:
            age = 37  # default is the mean of all other participants. already calculated.
        participantNumber = int(participantNumber)

        if isTraining:
            participantDataForRaceParticipation.append([
                year2012,
                year2013,
                year2014,
                year2015,
                gender,
                age,
                numberOfMontrealMarathons,
                distanceFromAverageAverage
            ])

            if attendedMontreal2015:
                participantDataForRaceTimes.append([
                    year2012,
                    year2013,
                    year2014,
                    year2015,
                    gender,
                    age,
                    numberOfMontrealMarathons,
                    distanceFromAverageAverage
                ])

            if attendedMontreal2015:
                montrealMarathonParticipaton.append(1)
                montrealMarathonTime.append(montreal2015FinishTime)
            else:
                montrealMarathonParticipaton.append(0)

        if not isTraining:
            participantDataFinalFor2016.append([
                year2013,
                year2014,
                year2015,
                year2016,
                gender,
                age,
                numberOfMontrealMarathons,
                distanceFromAverageAverage
            ])


    print participantDataForRaceTimes
    # Storing final array data.
#    pickle.dump(np.array(participantDataFinalFor2016), open('final_participantDataFinalFor2016.p', 'wb'))
    pickle.dump(np.array(participantDataForRaceParticipation), open('X_participantDataForRaceParticipation.p', 'wb'))
    pickle.dump(np.array(participantDataForRaceTimes), open('X_participantDataForRaceTimes.p', 'wb'))
    pickle.dump(np.array(montrealMarathonParticipaton), open('Y_montrealMarathonParticipaton.p', 'wb'))
    pickle.dump(np.array(montrealMarathonTime), open('Y_montrealMarathonTime.p', 'wb'))


def getAverageMarathonTimes(participants):
    marathonTotalTime = defaultdict(lambda: list())
    marathonAverageTime = dict()
    for raceList in participants.values():
        for race in raceList:
            if "marathon" in race[2].lower():
                if any(word in race[2].lower() for word in ["demi", "half"]):
                    raceName = strip_accents(race[1].lower()) + " demi"
                else:
                    raceName = strip_accents(race[1].lower())
                try:
                    finishTime = datetime.datetime.strptime(race[3], "%H:%M:%S")
                    finishSeconds = finishTime.second + finishTime.minute * 60 + finishTime.hour * 3600
                except:
                    finishSeconds = 0
                marathonTotalTime[raceName].append(finishSeconds)

    for race in marathonTotalTime:
        marathonTotalTime[race] = [max(marathonTotalTime[race]) if (x == -1) else x for x in marathonTotalTime[race]]
    for race in marathonTotalTime:
        marathonAverageTime[race] = sum(marathonTotalTime[race]) / len(marathonTotalTime[race])

    return marathonTotalTime, marathonAverageTime

def getFinishTime(finishTime, raceName, marathonTotalTime):
    """
    :param finishTime: provided string
    :param raceName: value if not -1, slowest time if -1.
    :return:
    """
    try:
        finishTime = datetime.datetime.strptime(finishTime, "%H:%M:%S")
        finishSeconds = finishTime.second + finishTime.minute * 60 + finishTime.hour * 3600
    except:
        finishSeconds = max(marathonTotalTime[raceName])

    return finishSeconds


def processAge(givenAge):
    """
    :param givenAge: string provided
    :return: age, gender
    """
    pattern1 = re.compile("^[A-Z][0-9]{2}\-[0-9]{2}$")
    pattern2 = re.compile("^[A-Z][0-9]{2}")
    pattern3 = re.compile("^[A-Z]\s[0-9]{2}\-[0-9]{2}$")
    pattern4 = re.compile("^M|H")
    pattern5 = re.compile("^F")
    if pattern1.match(givenAge):
        gender = givenAge[0]
        age = (int(givenAge[1:3]) + int(givenAge[4:6])) / 2
    elif pattern2.match(givenAge):
        gender = givenAge[0]
        age = (int(givenAge[1:3]))
    elif pattern3.match(givenAge):
        gender = givenAge[0]
        age = (int(givenAge[2:4]) + int(givenAge[5:7])) / 2
    elif pattern4.match(givenAge):
        gender = "M"
        age = None
    elif pattern5.match(givenAge):
        gender = "F"
        age = None
    else:
        gender = None
        age = None

    if gender == "F":
        gender = 0
    if gender == "M":
        gender = 1

    return age, gender


def strip_accents(s):
    """
    Thank you to http://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string for help
    :param s:
    :return:
    """
    s = s.decode('utf-8')
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

if __name__ == "__main__":
    main()
