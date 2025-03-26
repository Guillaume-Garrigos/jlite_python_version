import os, time, datetime
import multiprocessing

SESSION_LENGTH = 90 # in minutes
USED_TOKEN = multiprocessing.Value('i', 0) # a global variable. makes sure we call the dummy worker only once

def clock():# ensures time is set to Paris (UTC+1) wherever is run the notebook
    current_time = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    return current_time.strftime('%X') 

def dummy_worker():
    if USED_TOKEN.value:
        print("Vous ne pouvez maintenir une session ouverte qu'une seule fois")
    else:
        # makes sure the dummy woker is called only once
        USED_TOKEN.value = 1 
        # warn the user that they cannot run twice
        print("Session de " + str(SESSION_LENGTH) + " minutes commençant à " + clock() + ". Ne pas interrompre, ni relancer une deuxième fois.")
        # parameters of the pauses
        pause = 5
        chrono = 0
        nb_sleeps = int(SESSION_LENGTH/pause)
        for i in range(nb_sleeps):
            time.sleep(pause*60)
            chrono = chrono + pause
            print('Session ouverte depuis '+str(chrono)+' minutes', end="\r") # perdiodic refreshing
        # final message    
        print("Session de " + str(SESSION_LENGTH) + " minutes terminée à " + clock() + ".")
        print("Le serveur s'arrêtera après 10 minutes d'inactivité")
    return

def start_session():
    multiprocessing.Process(target=dummy_worker).start()