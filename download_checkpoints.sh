# on narigpu01 copy files to ~/public_html 
# make sure the 'other users' have rea and execute access to the files
# i.e. on narigpu in public_html run 
# chmod -R o+rx *
# below the link ~youraccountname/ is the toplevel of public_html


# download file from the url into the folder specified by the -P flag, relative to the official-phase-mins-eth directory
# copy this line for all files that need to be downloaded
wget -P TeamCode/src/work_dir  https://people.ee.ethz.ch/~huttercl/data/00000.zip 


# test by running 
# sh ./download_checkpoints.sh
#  from the official-phase-mins-eth directory 