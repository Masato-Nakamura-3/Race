# Race module for the Race model simulations
This is a module to simulate dynamic context-driven lexical activation and cloze responses, inspired by Staub et al. (2015) and Chen and Mirman (2012).
Use `help()` function to learn about what each function is doing and see `race_sample.ipynb` for a sample usage.



## Updates

9/26/2022

- Updated the default parameters in race_sample.ipynb based on Staub et al. (2015)'s experimental data.
- Updated the `race_fast` function to show progress in a better way.



12/13/2022

* Updated the `act2df` function and the `race_fast` function to better handle trials without any winners. Now there are two columns in the output dataframes, `cloze_prob` and `cloze_prob_sum1`. The `cloze_prob` in the output dataframe is calculated **including** such cases. That is, regardless of the number of trials without any winner, `cloze_prob` for a candidate that won 10 out of 50 trials is $10/50 = 0.2$. On the other hand, `cloze_prob_sum1` **excludes** those cases, and the value is $10 / (50 -10) = 0.25$ for the same candidate *if there are 10 trials without a winner*. (Therefore `cloze_prob_sum1` always sums up to 1, but `cloze_prob` doesn't.) 
