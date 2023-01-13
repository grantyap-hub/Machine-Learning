name: Grant Yap
email: gyap@u.rochester.edu


I did not finish this project.

To run my program, I didn't implement the argparse as I didn't have enough time so you would need to look
into hmm.py to run through the get_alpha, get_beta, get_gammas, and e_m_step functions I created within 
the HMM class. My expt class is empty and I was not able to write a report due to this.

In theory, I believe I have implemented the alpha, beta, digamma, and part of the
e_m step of the forward backward algorithm described in https://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf
provided by the class.

I implemented my functions based on the pseudocode provided by the document and believe what I have to be 
correct.

Data Loading:
At the moment, my program was only run on small cases to the test the algorithms. For example, I had changed
the load_subdir function to only take in as many functions as an integer passed into the function.
(At time of submission this would be 10 located at line 208 of the code. But if this variable was changed to
a larger number, I have tested it out with all 50,000 files and it takes approximately 5 minutes to load all files
in.

Vocabulary building:
I defined my vocabulary to be made up of only 28 unique characters (white space, exclamation mark, and the 
rest of the alphabet). Using regex expressions, I was able to reduce each file into one-hot-enodings that 
would be processed into the completed model had I successfully built the project.

Instantiating HMM (and subsequent steps):
I was not able to finish this step and subsequent calculations. What I had been able to do as mentioned before, was
to calculate alpha, beta, gammas, and part of e_m step. While I was not able to test this formally, I could see based
on my saved c values that the log likelihood was at the very least in a reasonable range of approximately -2500. I also
believe my logic to be correct as this was directly translated from "A Revealing Introduction to Hidden Markov Models".

I am submitting this project as the deadline is tonight, but I would also like to mention that I will try to finish
the project in my own time before the end of the week. If it is possible to get partial credit for the work that I 
have done, that would be appreciated.

