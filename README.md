# Race module for the Race model simulations
This is a module to simulate dynamic context-driven lexical activation and cloze responses based on Staub et al. (2015), and is refined reagarding Nakamura and Phillips (2022) and Chen and Mirman (2012).

You can use this to generate speeded cloze data, or you can modify the functions to develop your own model.



## About the model

In the speeded cloze task, participants are asked to provide continuations to sentence fragments, and the responses and their latencies are recorded. Both the frequency of each lexical item (i.e. cloze probability) and the latency (i.e. reaction time / RT) are used as prediction measures. When a lexical item has a high cloze probability and a short RT, it is considered that it is highly predictable given the context. 

Th Race model is a linking hypothesis for this task, connecting the underlying predicability/strength of each lexical item to the observable measures (RTs) through parallel activation of competing lexical items (See Staub et al. (2015) for empirical supports for the model). In the Race model, each competitor (lexical item) accumulates activation, and the first comeptitor that reaches a threshold is produced. 

This computational model simultaes this Race process in a step-by-step manner, where each item gains or loses activation in each cycle. In each cycle, each competitor gains activation depending on its own strength. More specifically, each competitor has its own normal distribution, and the amount of activation it gains is deterimined by random sampling from the distribution. The means of the normal distributions, or the "strengths" of the competitors, are the µ parameters of this model. The variability of the normal distribution is determined by the σ parameter. (The model assumes that the variability of the normal distributions are constant, but you could assume otherwise.)

Competitors also inhibit each other in each cycle. By default, the amount of inhibition each comeptitor imposes on other competitors is determined by a sigmoid function (Chen & Mirman (2012) provides supports for this).

The competitors thus repeat gaining activation and receiving inhibition, but once one of the competitor reaches a threshold, it is produced.

## Installation

Please download the entire folder and place it whereever you want. Make sure to add the path to the the Race folder using `sys.path.append(Path_To_Module)` function before importing race.py.



## Functions

`activate`
 This function randomly generates activation from normal distributions. This is mainly used for other functions.

`inhibit_loss`
This function computes the amount of inhibition each competitor receives. This is also used for other functions.

`single_race`
This function simulates a single race from beginning to the end.

`race`
This function simulates multiple races with different sets of competitors. It keeps track of the activation of each competitor at each cycle in each race, and output the entire record.

`act2df`
This function maps activation patterns onto two sets of cloze data. One keeps track the winner of each race, and the other aggregates the races and provide information such as cloze probabilities for each competitor.

`race_fast`
This function basically does the same thing as `race` and `act2df`, but it doesn't give you the record of activation patterns and it only gives you the resulting cloze data. Instead, it runs much faster and is suitable for simulation of a lot of races (e.g. 40 races for 500 sets of competitors).

Use `help()` function to learn more about what each function is doing and see `race_sample.ipynb` for a sample usage.

## Updates

1/5/2023

* Added more detailed descriptions of the model in README.



12/13/2022

* Updated the `act2df` function and the `race_fast` function to better handle trials without any winners. Now there are two columns in the output dataframes, `cloze_prob` and `cloze_prob_sum1`. The `cloze_prob` in the output dataframe is calculated **including** such cases. That is, regardless of the number of trials without any winner, `cloze_prob` for a candidate that won 10 out of 50 trials is $10/50 = 0.2$. On the other hand, `cloze_prob_sum1` **excludes** those cases, and the value is $10 / (50 -10) = 0.25$ for the same candidate *if there are 10 trials without a winner*. (Therefore `cloze_prob_sum1` always sums up to 1, but `cloze_prob` doesn't.) 



9/26/2022

- Updated the default parameters in race_sample.ipynb based on Staub et al. (2015)'s experimental data.
- Updated the `race_fast` function to show progress in a better way.
