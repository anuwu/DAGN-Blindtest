# DAGN-Blindtest
Secondary stage of the DAGN (Dual Active Nuclei Galaxy) project. This code uses the previously developed algorithm to detect previously DAGNs from a list of Galaxy IDs from SDSS.

Changes from the previous repository -
1. Uses only environment peak search algorithm. Deleted the $std sized image processing parts.
2. Did away with the init.py module. 
3. Consistent dimensions of images generated in all intermediate steps.
4. Implemented SNR. Added another class called - 'NoPeak' - apart from the regular 'Single' or 'Double'.
5. Added feature to send e-mail notifications for completion progress.

The pipeline has been designed to run on the Google Colab environment. A lot of downloads is required for which Google's network will prove useful. Also there are form-fields that help to setup some variables necessary for running the pipeline. If you wish, you could also run it locally. Just ensure to assign legal values to the form variables.

Steps to run the pipeline -
1. In one way or another, clone the repository into a directory in your google drive. Note down this directory.
2. Create another directory that contains a .csv file of the objIDs of galaxies that you want to test upon. Also note this directory.
3. Open the notebook and run as pe the instructions written in the notebook.

Steps to query SDSS to obtain galaxy objIDs -
1. Create an account on http://www.sciserver.org/ if you don't already have one.
2. Send an e-mail to dagn2020iia@gmail.com with your SciServer username. I shall add you to the collaborative group that contains the master table of all galaxy. The name of the group is AstrIRG_DAGN.
3. Open CasJobs and execute any query on SDSS DR16's view named 'Galaxy'. Ensure to insert the following phrase at the appropriate location in your query.

> NOT IN (SELECT * FROM AstrIRG_DAGN.dagn2020.SDSSDR16_Galaxy_Master)

This will ensure that no objID, that has already been processed previously, ends up in your query resut.
4. Download the query result as a .csv file and place it in the appropriate location in your google drive. Now you can run the pipeline

Kindly send any queries regarding the project to f2016590@pilani.bits-pilani.ac.in or dagn2020iia@gmail.com
Happy Searching!
