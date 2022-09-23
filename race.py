#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

def activate(mulist, sigma):
    """Randomly generate activation from normal distributions. This is used for other simulations.

    Parameters
    ----------
    param1 : mulist
        A list of mu's of the normal distributions. Each value of the list represents each competitor.
    param2 : sigma
        A single value of the sigma parameter of the normal distributions.

    Returns
    -------
    act
        Activation of each competitor.
    """
    ncomp = len(mulist)
    act = np.random.normal(0, sigma, ncomp) + mulist

    return act


def inhibit_loss(prev_act, rate, sigmoid = True, beta = 35, x_zero = 0.5):
    """Compute linear or sigmoid inhibition from activation.

    Parameters
    ----------
    param1 : prev_act
        A numpy array of accumulated activation of each participant.
    param2 : rate
        The rate of the inhibition.
    param3 : sigmoid
        Whether the inhibition is a sigmoid or a linear function.
    param4 : beta
        Beta parameter of the sigmoid function. (How steep it is.)
    param5 : x_zero
        X_zero parameter of the sigmoid function. (Where it gets steeper.)

    Returns
    -------
    loss
        Amount of activation each competitor lost due to inhibition.
    """

    # Calculate activation lost by inhibition
    if sigmoid == True: # If non-linear (sigmoid)
        inh_coef = 1 / (1+np.exp(-beta*(prev_act - x_zero))) # sigmoid function
    else:
        inh_coef = 1 # If linear

    inh_array = rate * inh_coef * prev_act #sigmoid function

    # Each competitor's loss is the some of inhibition from other competitors minus its own inhibition
    loss =  np.sum(inh_array) - inh_array

    return loss

def single_race(mu, sigma, max_cycle = 100, threshold = 0.7, init = 0.1, rate = 0.2, sigmoid = True, beta = 35, x_zero = 0.5, cm = True, decay = 0):
    """Simulate a single race.

    The simulation continues until either the number of the cycles reaches the upper limit or one of the competitors' activation reaches the threshold. Assign a value grater than 1 to threshold to keep running the race until the maximum number of cycles.

    Parameters
    ----------
    param1 : mu
        A numpy array of mu's of the normal distributions. Each value of the represents each competitor.
    param2 : sigma
        A single value of the sigma parameter of normal distributions.
    param3 : max_cycle
        Maximum number of cycles in each race. Default is 100.
    param4 : threshold
        The threshold of activation to decide the winner. Assign a value grater than 1 to keep running the race until the maximum number of cycles.
    param5 : init
        An integer or a numpy array specifying the initial values of the activation.
    param6 : rate
        The rate of the inhibition.
    param7 : sigmoid
        Whether the inhibition is a sigmoid or a linear function. Default is True.
    param8 : beta
        Beta parameter of the sigmoid function. (How steep it is.)
    param9 : x_zero
        X_zero parameter of the sigmoid function. (Where it gets steeper.)
    param10 : cm
        Whether to use Chen and Mirman's weighting of activation. Default is True.
    param11 : decay
        The rate of the decaying. Default set to zero.

    Returns
    -------
    acti
        A numpy array of the amount of activation of each competitor at each cycle.
    """
    ncomp = len(mu)

    #Generate activation gained in each cycle
    actmat = np.random.normal(0, sigma, [max_cycle, ncomp]) + np.tile(mu, [max_cycle, 1])

    # Set initial values. If an integer is passed, copy that and create an array.
    # If an array is passed, use that as the initial values.
    if (type(init) == int) | (type(init) == float):
        acti = np.tile(init, [1, ncomp])
    elif len(init) == ncomp:
        acti = init.reshape(1, ncomp)
    else:
        print("Something is going wrong with the initial values!")

    i = 0
    # Loop until the maximum iteration or something reach the threshold
    # You can assign a threshold value greater than 1 if you want to keep it running until some iteration
    while (i < max_cycle) & (np.max(acti[-1]) < threshold):

        #new activation - inhibition
        tacti = actmat[i] - inhibit_loss(acti[-1], rate, sigmoid = sigmoid, beta = beta, x_zero = x_zero)

        if cm == True:
            # Chen and Mirman's weighting function and decay.
            nacti = acti[-1] + np.where(tacti > 0, (1 - acti[-1]) * tacti, acti[-1] * tacti)  - np.tile(decay, ncomp) * acti[-1]
        else:
            nacti = acti[-1] + tacti


        # Activation does not become negative or exceed 1
        nacti[nacti < 0] = 0
        nacti[nacti > 1] = 1


        acti = np.concatenate([acti, nacti.reshape(1, ncomp)])

        i += 1

    return acti




def race(mu, sigma, nrace, max_cycle = 100, threshold = 10, init = 0.1, rate = 0.2, sigmoid = True, beta = 35, x_zero = 0.5, cm = True, decay = 0):
    """Simulate multiple races for a single set of candidate and output activation patterns. (i.e. One context with multiple participants)

    The simulation continues until either the number of the cycles reaches the upper limit or one of the competitors' activation reaches the threshold. By default a large value (10) is assigned to the threshold parameter to keep running the race until the maximum number of cycles, and it is recommended to keep that unless there is any reason not to do otherwise.

    Parameters
    ----------
    param1 : mu
        A numpy array of mu's of the normal distributions. Each value of the represents each competitor. This array could be 1-dimension (for a single context), or 2-dimension (for multiple races), where the Axis 0 represents contexts and Axis 1 represents competitors.
    param2 : sigma
        A single value of the sigma parameter of normal distributions.
    param3 : nrace
        The number of races to simulate (or, the number of participants)
    param4 : max_cycle
        Maximum number of cycles in each race. Default is 100.
    param5 : threshold
        The threshold of activation to decide the winner. Assign a value grater than 1 to keep running the race until the maximum number of cycles.
    param6 : init
        An integer or a numpy array specifying the initial values of the activation.
    param7 : rate
        The rate of the inhibition.
    param8 : sigmoid
        Whether the inhibition is a sigmoid or a linear function. Default is True.
    param9 : beta
        Beta parameter of the sigmoid function. (How steep it is.)
    param10 : x_zero
        X_zero parameter of the sigmoid function. (Where it gets steeper.)
    param11 : cm
        Whether to use Chen and Mirman's weighting of activation. Default is True.
    param12 : decay
        The rate of the decaying. Default set to zero.

    Returns
    -------
    actmat_all
        A numpy array of the amount of activation. Axis 0: trial; axis 1: cycle; axis 2: competitor.
    """

    # Reshape mu and create a two dimension array (Context x Competitor)
    if mu.ndim == 1:
        mu = np.reshape(mu, [1, len(mu)])
    elif mu.ndim > 2:
        print("Too many dimensions for mu")

    ncontext = np.shape(mu)[0]
    ncompetitor = np.shape(mu)[1]

    # Generate slots to store activation
    actmat_all = np.zeros([ncontext, nrace, max_cycle + 1, ncompetitor])

    # Loop for each context
    for i in range(0, ncontext):

        # Loop for each trial
        for j in range(0, nrace):
        # Get the activation for one trial / race
            temp_actmat = single_race(mu[i], sigma, max_cycle = max_cycle, threshold = threshold, init = init, rate = rate, sigmoid = sigmoid, beta = beta, x_zero = x_zero, cm = cm, decay = decay)
            # Replace the slots with the generated activation
            actmat_all[i, j] = temp_actmat



    return actmat_all

def act2df(activation, threshold):
    """Turn a 4-dimension activation matrix for a single context (Context x Trial x Cycle x Competitor) to an analyzable dataframe.

    The first output data frame shows the results of all races, and the second one shows the summary of each type of responses in each context.
    Note that when no item reaches a threshold in a trial, no data will be generated for that. (i.e. No winner)

    Parameters
    ----------
    param1 : activation
        A 4-dimension numpy array (Context x Trial x Cycle x Competitor).
    param2 : threshold
        The threshold of activation to decide the winner.

    Returns
    -------
    df_trial
        A pandas data frame with three columns. "context_id" specifies the id of the context, "ft" specifies the cycle where the first item reaches the threshold, "response" specifies the id of the winner competitor, and "rt_25" is the reaction times.
    df_summary
        A pandas data frame that summarizes df_trial. "total_count" represents the number of races in each context, "count" is the number of wins by each competitor, "ft" is the average finishing time.
    """
    # Find all activation greater than the threshold and the context, race, cycle, and competitor information of those situations
    context, race, cycle, competitor = np.where(activation > threshold)
    # Find the indices of the first cycles where one item reached the threshold in each race for each context
    th_idx = [np.where((context == i) &  (race == j))[0][0] for i in range(0, np.shape(activation)[0]) for j in range(0,np.shape(activation)[1])]

    df_trial = pd.DataFrame({"context_id": context[th_idx], "response":competitor[th_idx], "ft":cycle[th_idx]})
    # Compute reaction times based on assumptions: (i) each cycle is 25ms, and (ii) it takes 350ms for non-lexical processes for production
    df_trial["rt_25"] = df_trial["ft"] * 25 + 350

    # Generate the summary
    average_ft = df_trial.groupby(["context_id", "response"], as_index = False).mean()
    total_count = df_trial[["context_id", "ft"]].groupby(["context_id"], as_index = False).count().rename(columns = {"ft": "total_count"})
    win_count = df_trial[["context_id", "response", "ft"]].groupby(["context_id", "response"], as_index = False).count().rename(columns = {"ft": "count"})

    df_summary = pd.merge(pd.merge(average_ft, total_count), win_count)

    # Compute cloze probabilities
    df_summary["cloze_prob"] = df_summary["count"] / df_summary["total_count"]

    # Get modal cloze probs
    m_cloze = df_summary[["context_id", "cloze_prob"]].groupby(["context_id"], as_index = False).max().rename(columns = {"cloze_prob": "modal_prob"})
    df_summary = pd.merge(df_summary, m_cloze, how = "left")

    # Import the cloze data back to the trial data
    df_trial = pd.merge(df_trial, df_summary[["context_id", "response", "cloze_prob", "modal_prob"]], how = "left")

    return df_trial, df_summary




def race_fast(mu, sigma, nrace, max_cycle = 100, threshold = 0.7, init = 0.1, rate = 0.2, sigmoid = True, beta = 35, x_zero = 0.5, decay = 0):

    # Basically the same as other race + act2df, but runs faster without outputting activation patterns

    ncontext = np.shape(mu)[0]
    ncompetitor = np.shape(mu)[1]

    context = []
    winner = []
    ft = []

    # Progress
    ten_pcs = [round(nc * ncompetitor / 10) for nc in range(1, 10)]
    pcs = 10



    # Reshape mu and create a two dimension array (Context x Competitor)
    if mu.ndim == 1:
        mu = np.reshape(mu, [1, len(mu)])
    elif mu.ndim > 2:
        print("Too many dimensions for mu")

    # Set initial values. If an integer is passed, copy that and create an array.
    # If an array is passed, use that as the initial values.
    if (type(init) == int) | (type(init) == float):
        initial = np.tile(init, ncompetitor)
    elif len(init) == ncompetitor:
        initial = init.reshape(ncompetitor)
    else:
        print("Something is going wrong with the initial values!")

    #Generate activation gained in each cycle
    actmat = np.random.normal(0, sigma, (ncontext, nrace, max_cycle, ncompetitor)) + np.tile(mu.reshape((ncontext, 1, 1, ncompetitor)), (1, nrace, max_cycle, 1))

    # Loop for each context
    for i in range(0, ncontext):
        # Print progress
        if i in ten_pcs:
            print(f'{pcs}% Completed.')
            pcs += 10

        # Loop for each trial
        for j in range(0, nrace):

            # Set initial values
            acti = initial
            

            n = 0
            # Loop until the maximum iteration or something reach the threshold
            # You can assign a threshold value greater than 1 if you want to keep it running until some iteration
            while (n < max_cycle) & (np.max(acti) < threshold):

                #new activation - inhibition
                tacti = actmat[i, j, n] - inhibit_loss(acti, rate, sigmoid = sigmoid, beta = beta, x_zero = x_zero)
                nacti = acti + np.where(tacti > 0, (1 - acti) * tacti, acti * tacti)  - np.tile(decay, ncompetitor) * acti

                # Activation does not become negative or exceed 1
                nacti[nacti < 0] = 0
                nacti[nacti > 1] = 1

                acti = nacti.reshape(ncompetitor)
                n += 1

            context.append(i)
            winner.append(np.argmax(acti))
            ft.append(n)

    df_trial = pd.DataFrame({"context_id": context, "response":winner, "ft":ft})
    # Compute reaction times based on assumptions: (i) each cycle is 25ms, and (ii) it takes 350ms for non-lexical processes for production
    df_trial["rt_25"] = df_trial["ft"] * 25 + 350

    # Generate the summary
    average_ft = df_trial.groupby(["context_id", "response"], as_index = False).mean()
    total_count = df_trial[["context_id", "ft"]].groupby(["context_id"], as_index = False).count().rename(columns = {"ft": "total_count"})
    win_count = df_trial[["context_id", "response", "ft"]].groupby(["context_id", "response"], as_index = False).count().rename(columns = {"ft": "count"})

    df_summary = pd.merge(pd.merge(average_ft, total_count), win_count)

    # Compute cloze probabilities
    df_summary["cloze_prob"] = df_summary["count"] / df_summary["total_count"]

    # Get modal cloze probs
    m_cloze = df_summary[["context_id", "cloze_prob"]].groupby(["context_id"], as_index = False).max().rename(columns = {"cloze_prob": "modal_prob"})
    df_summary = pd.merge(df_summary, m_cloze, how = "left")

    # Import the cloze data back to the trial data
    df_trial = pd.merge(df_trial, df_summary[["context_id", "response", "cloze_prob", "modal_prob"]], how = "left")
    print("Done.")

    return df_trial, df_summary



# I am not using this function anymore
# simulate races for a single context until an item wins and get a data frame
# Whe no competitor wins, the one with the largetst activation at the last cycle is treated as the winner
def single_context_complete(mu, sigma, nrace, max_cycle = 100, threshold = 0.7, init = 0.1, rate = 0.2, sigmoid = True, beta = 35, x_zero = 0.5, cm = True, decay = 0):

    fts = []
    winners = []

    for i in range(0, nrace):
        # Get the activation for one context
        actmat = race.single_race(mu, sigma, max_cycle = max_cycle, threshold = threshold, init = init, rate = rate, sigmoid = sigmoid, beta = beta, x_zero = x_zero, cm = cm, decay = decay)


        fts.append(actmat.shape[0] - 1) # Nth cycle (initial one is 0th cycle)
        winners.append(np.argmax(actmat[-1]))

    df = pd.DataFrame({"winner": winners, "ft": fts})

    return df
