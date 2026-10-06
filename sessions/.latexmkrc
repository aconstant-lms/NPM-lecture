# The sessions are compiled from this directory but read the shared
# preamble, the code files and the labels of the book (../main.aux)
# with paths relative to the repository root: put the root on TEXINPUTS.
$ENV{'TEXINPUTS'} = '.:..:' . ($ENV{'TEXINPUTS'} // '');
$pdf_mode = 1;
