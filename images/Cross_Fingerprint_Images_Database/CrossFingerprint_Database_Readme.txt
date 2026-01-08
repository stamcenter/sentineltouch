Database Description - The Hong Kong Polytechnic University Contactless 2D to Contact-based 2D Fingerprint Images Database
=======================================================

This two-session cross fingerprint dataset is organized into two main folders, i.e. contactless 2D fingerprints and conresponding contact-based fingerprints. 
This is a two-session database and in each of the session, the contactless 2D fingerprint images and 'corresponding' contact-based fingerprints images acquired from the volunteers during the Sep 2014 to Feb 2017 period using 
a digital CMOS camera and URU fingerprint reader. This database contains 2976 contatcless 2D fingeprint images and conresponding 2976 contact-based fingeprint from 336 clients. Six contactless and contact-based fingerprint
images (impressions) were acquired from each finger. The entire database is released to advance much needed research efforts. However only the first session data from 300 clients was acquired during the first stage and used for the work 
described in the paper [1]. 

Size of Database
========================
The size of each contatcless 2D fingerprint image is around 3.60 MB. The size of each contact-based fingerprint is around 64.00 KB. The size of each downsampled contatcless 2D fingerprint image is around 79.00 KB.
The size of the whole database is around 10.8 GB. 

Platform
========================
The database contains contatcless 2D fingerprint images which are stored in bitmap and contact-based fingerprints which are stored in JEPG format. The fingerprint images data can be opened and processed in VC++ or MatLab.

Organization of Database
========================
This database contains 2976 contactless 2D fingerprint images and conresponding contact-based fingerprints. 
The first session part of database was acquired from 336 different clients/fingers. Each of the client provided 6 different fingerprint samples (6 images).  The second session part of the database contains images 
from corresponding 160 clients, and each of these second-session clients provided 6 fingerprint samples (6 images) after an interval of 2-24 months.  Therefore, there a total of 5952 images were acquired for this
database and accessible from two different folders.

Every six contactless 2D fingerprint images are in folder 'contactless_2d_fingerprint_images' named as 'pX/pY', where X represents user finger identification number, Y represents image sample number. The resolution of each image is 1400*900.
Every six contact-based fingerprint are named as 'X_Y' in folder 'contact-based fingerprints', where X represents user finger identification number, Y represents image sample number. The resolution of each image is 328*356. 
The downsampled grayscale contactless fingerprint images (used in cited reference) are in folder 'processed_contactless_2d_fingerprint_images' named as 'pX/pY', where X represents user finger identification number, Y represents image sample number.

Usage of Database
========================
This database is only available for research and noncommercial purposes. All the rights of "The Hong Kong Polytechnic University Contactless 2D to Contact-based 2D Fingerprint Images Database" are reserved and commercial use/distribution of this 
database is strictly prohibited. All the technical reports and papers that report experimental results using this database must provide due acknowledgement and citation to reference [1] in the following. 

Related Publication:
====================
[1] Chenhao Lin, Ajay Kumar, "Matching Contactless and Contact-based Conventional Fingerprint Images for Biometrics Identification," IEEE Transactions on Image Processing, vol. 27, pp. 2008-2021, April 2018.

Contact Information:
====================
Dr. Ajay Kumar
Department of Computing. The Hong Kong Polytechnic University
Hong Kong
E-mail: Ajay.Kumar@polyu.edu.hk