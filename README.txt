---------------------------------------------------------------
Calibration code for 'Descending Price Optimally Coordinates Search'
by Kleinberg, Waggoner, and Weyl.

This version: August 2016

Contact: Bo Waggoner <bwaggoner@fas.harvard.edu>
---------------------------------------------------------------

Abstract: This code simulates auctions in two settings: a "startups" model
and the "timber" model of Roberts & Sweeting, American Economic Review 2013.
It computes the welfare (and revenue) of several kinds of auctions.
The main tool is the programming language python; we also used Katzwer's
AuctionSolver
    <https://www.princeton.edu/~rkatzwer/AuctionSolver/>
and Roberts-Sweeting's code:
    <https://www.aeaweb.org/articles?id=10.1257/aer.103.5.1830>

Our results were produced with the above tools and the below code
written in Python version 3.5.1, using the numpy, scipy, and (optional)
matplotlib libraries, running on Linux.

Note: code uses python 3, probably not compatible with python 2.

-------------------------------
List of files and folders
-------------------------
check-startup-equil.py
    Produces plots to visually sanity-check the equilibrium output
    of startups_equil.py.

get-startups-equil.py
    Compute equilibrium and parameters for startups model.
    Produces output in file 'startups_equil.py'.

get-startups-results.py
    Compute welfare and revenue using the equilibrium computed in
    'startups_equil.py'. Produces output in file 'startups_results.py'.

get-timber-equil.py
    Computes parameters for timber; produces 'timber_equil.py'.

get-timber-results.py
    Computes welfare for timber; produces 'timber_results.py'.

startups_equil.py
startups_results.py
timber_equil.py
timber_results.py
    Files produced by the above "get" scripts. Contain all the data/results.
    Running 'startups_results.py' and 'timber_results.py' prints out
    and produces plots of the results as well.

tools/
    Contains the main python code that is used by the above scripts.


-------------------------------
Startups Instructions
---------------------
We used the following steps to produce our results.
1. Manually edit 'get-startups-equil.py' to choose the parameters for
   each scenario. A scenario consists of a set of bidder parameters and
   the number of bidders in the auction.

   Note: to replicate a scenario that was already generated and run, just edit
   'get-startups-equil.py' by copy/pasting kinds_params_list and ns_list into
   the file instead of generating new ones; see the code for details.

2. Run 'get-startups-equil.py', e.g. on the command line:
      python3 get-startups-equil.py
   Edit parameters to control computational intensiveness.
   On my machine with num_bid_pts = 1000, num_mus = 1000,
   takes 1-2 hours to run (utilizing 4-8 cores) and uses ~1-2 GB of RAM.

   For each scenario, this generates a number of asymmetric bidders according to
   the parameters, fits a lognormal distribution to each of their covered-call
   value distributions, and attempts to solve for equilibrium of a simultaneous
   second-price auction. Produces output file 'startups_equil.py'.
   Also prints some updates on progress along the way.

3. For each scenario, use Katzwer's AuctionSolver to solve for equilibrium of
   a first-price auction where value distributions match the lognormal fitted
   covered-call distributions. Manually edit the file 'startups_equil.py' to
   paste in the results. Details are given below in (*).

   (Optional.) At this point, you can also run 'check-startups-equil.py' to
   visually sanity-check the equilibrium and lognormal fits.

4. Run 'get-startups-results.py'. May take some hours and uses multiple
   CPU cores and a lot of RAM; manually edit to try small parameters first.
   Note that it takes some time to get going (about 12min on my 4+ core
   machine) because it is pre-computing a bunch of useful info before
   starting the auctions.

   For each scenario, computes optimal welfare (average maximum covered-call
   value) and the welfare/revenue of the simultaneous auction and the
   descending-price auction. Produces output file 'startups_results.py'.

5. The results can be viewed by running 'startup_results.py',
   which prints the results and gives some plots.
   One can also open the file itself to inspect the results.


(*) Using AuctionSolver:
    Launch AuctionSolver.jar, obtained from Katzwer's website, using java,
    e.g. on the command line:
        java -jar AuctionSolver.jar
    Then, for each scenario, we enter value distributions for all bidders,
    ask AuctionSolver to find an equilibrium, and copy/paste that equilibrium
    back into 'startups_equil.py'. Details follow.

    In the file 'startups_equil.py', the fitted lognormal distributions'
    (mu, sigma) are stored in the variable fit_params_list. However, because
    AuctionSolver requires mu > 0, we will instead use "shifted" versions
    where all mu in a scenario have been shifted by the same constant. This
    scales all distributions by a constant, which does not change strategic
    behavior. These shifts are stored in the variable solver_mu_shifts, but
    we don't need it now.

    We need the variable solver_shifted_params_list. It looks like
        solver_shifted_params_list = [
          # scenario 0
          [
            [3, 1],
            [4, 1],
            [3, 2],
          ],

          # scenario 1
          [
            [2, 2],
            [2, 1],
          ],
        ]
    In this case, scenario 0 has three bidder "kinds" whose (scaled)
    covered-call distributions are approximately lognormal with mu=3 and
    sigma=1 for the first bidder, mu=4 and sigma=1 for the second, etc.
    Scenario 1 has two bidder kinds, etc.

    For each scenario, repeat the following procedure.
    Begin by opening AuctionSolver and selecting "Asymmetric N-bidder Solver".

    A. Choosing bidder types.
    Select "Choose bidder types". For each bidder kind in the scenario,
    use "Write New Distribution(s)". Enter a name for this bidder, e.g.
    'scenario_0_bidder_0'. Choose "Parameterize a pre-coded distribution".
    Select lognormal; it asks for four parameters. The first two, omegaL and
    omegaH, are the lower and upper bounds to truncate the distribution to.
    We use 0 for omegaL and pick OmegaH somewhat arbitrarily "high enough".
    The variable solver_upper_bounds, in the file startups_equil.py, contains
    some suggested upper bounds to use.

    We use the same OmegaH for all bidders in a given scenario.
    Then copy and paste the bidder's mu and sigma from fit_params_list.
    After choosing "OK", choose "Save Distribution".
    See troubleshooting in (**) if a problem arises here.

    B. Solving for equilibrium.
    After creating all the bidders in a scenario, in the "Make a Distribution
    Profile" screen, select the bidders just created from the left-side
    "Distributions Library" menu and choose "Add to Profile". IMPORTANT:
    Make sure they are listed in the same order as they appear in
    fit_params_list! Then choose "OK". Choose the number of each
    kind of bidder. In our startups simulations, there is always one of
    each kind.

    AuctionSolver has several solution methods (see its dropdown menu of that
    name); we use the default "Fixed Point Finite-Difference Iteration".
    You do not need to do anything to select this as it is already the default.
    We go to the "Numerical Parameters" dropdown menu and select the top
    option, "Set Fixed-Point Iteration Parameters". We leave all parameters
    at default except for choosing Grid Size to be 1000.
    Choose "OK", then in the main screen, "Solve for Bayesian-Nash
    equilibrium". See troubleshooting in (**) if a problem arises here.

    If solving was successful, go to the "Data Printouts" tab. Then from the
    dropdown menu "Data", choose "Print Inverse Bid Functions".
    ****KEY STEP****:
    Copy and paste the output numbers into the file 'startups_equil.py',
    formatted as follows. (Note: we used the vim text editor which made the
    following procedure pretty easy. You may wish to use some tools to help.)
    At the end of the 'startups_equil.py' file is the variable solver_output_fp
    which looks about like this:
        solver_output_fp = [
          # scenario 0
          [],
          # scenario 1
          [],
          ]
    In this example, we have two scenarios. We will paste all the numbers from
    the data printout for the first scenario in between the square brackets for
    the first scenario and add commas between all the numbers.
    Then do the same for the second scenario. The result looks something like:
        solver_output_fp = [
          # scenario 0
          [0.0010000000,  0.0010000000,   0.0010000000,   0.0010000000,
           0.2873201618,  0.3009998000,   0.3019598013,   0.3010872131],

          # scenario 1
          [0.0010000000,  0.0010000000,   0.0010000000,   0.0010000000,
           0.2873201618,  0.3009998000,   0.3019598013,   0.3010872131],
          ]
    But with a lot more lines. Line breaks and whitespace are optional. The
    numbers must be in the same order and comma-separated.

    Explanation: the first column from AuctionSolver is a list of discretized
    bids; the next columns are a list of corresponding values for each bidder in
    equilibrium. So with three bidders, our code reads the input for that
    scenario in groups of four (bid, bidder_value, bidder_value, bidder_value).
    
    If unable to get the results for a scenario, leave it as an empty list
    like ths:
        solver_output_fp = [
          # scenario 0
          [0.0010000000,  0.0010000000,   0.0010000000,   0.0010000000,
           0.2873201618,  0.3009998000,   0.3019598013,   0.3010872131],

          # scenario 1
          [],

          # scenario 2
          [0.0010000000,  0.0010000000,   0.0010000000,   0.0010000000,
           0.2873201618,  0.3009998000,   0.3019598013,   0.3010872131],
          ]solver_output_fp = [

    Now 'startups_equil.py' is complete and we are ready to run
    'get-startups-results.py'.

    Final note: due to the mu shift, these values we pasted into
    'startups_equil.py' will need to be scaled by e^{shift}. Our code in
    'get-startups-results.py' will do this automatically using the shifts
    stored in the variable solver_mu_shifts.
    


(**) Troubleshooting AuctionSolver.
     A. We sometimes had problems saving/compiling lognormal value
     distributions with higher variance. One option if possible is to
     further shift down the mu parameter of each variable.
     Unfortunately, sometimes this was not enough and we had to re-draw
     the scenario.

     B. AuctionSolver may have trouble finding an equilibrium. We first
     tried reducing the grid size from 1000 down as far as 250. If it
     still fails, including fiddling with the other parameters, we
     sometimes had success by re-entering the bidder distributions,
     but with the upper bound omegaH smaller and close to the visually
     apparent reasonable upper bound on the lognormal CDFs.


-------------------------------
Timber Instructions
---------------------
We used the following steps to produce our results.
1. Re-run Roberts and Sweeting's code for each scenario from Table 4 and 5 of
   their paper, to replicate these tables and get the welfare of the
   simultaneous and sequential mechanisms.

   Note 1: at the time we accessed their code, we communicated with the authors
   to make some small changes in order to exactly replicate their results.
   The main change was modifying the grid variable to:
      grid=(0.01:0.01:500)'
   At the time you read this, they may have updated the code with this change
   included.

   Note 2: it may take several days total of computing time to replicate
   all of the scenarios in the table.
 
2. Manually set parameters in file 'get-timber-equil.py' to match Roberts
   and Sweeting, Tables 4 and 5. Run it to produce file 'timber_equil.py'.

3. Exactly as described above for startups, use AuctionSolver to solve
   for first-price equilibrium using the lognormal covered-call value
   distributions in 'timber_equil.py', and (KEY STEP) write the results into
   the variable solver_output_fp at the end of 'timber_equil.py'.

   Difference 1: We didn't automatically shift the mu parameters of the
   covered-call distributions. In a couple of instances where AuctionSolver
   had trouble, we manually shifted the mus, then manually scaled up
   the output of AuctionSolver.

   Difference 2: is that there are always two kinds of bidders (loggers
   and mills, in that order). So each scenario only has two bidder value
   distributions (and they should be kept in the same order as the parameters
   in 'timber_equil.py'). However, make sure to choose the correct number of
   each kind of bidder, e.g. in the baseline scenario there are 4 of each.
   For the scenario with only one kind of bidder, we did not run it because
   we know, since all bidders are symmetric, that descending-price achieves
   optimal welfare.

4. Run 'get-timber-results.py' to produce output file 'timber_results.py'.
   May take some time to run; maybe start with smaller parameters.

5. View 'timber_results.py' directly, or run it to print and plot results.


