# CAPTCHA_cracking

use Deep learning to crack CAPTCHA

## What is CAPTCHA

CAPTCHA was short for Completely Automated Public Turing test to tell Computers and Humans Apart.

It is a program intended to distinguish human from machine input, 
typically as a way of thwarting spam and automated extraction of data from websites.

## What we do

We use a CNN to classify a CAPTCHA image input to a multi-digit catrgory in a single run. 

## What is good in it.

Use CNN to classify not a digit at a time but all digits in the same time. This was made via a carefully designed softmax.

Use active learning to improve the generalizing performance.
