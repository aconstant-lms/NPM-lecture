# The companion is compiled from this directory but reads the shared preamble,
# the cast3m/ files and the labels of the book (../main.aux, ../chapters/*.aux)
# with paths relative to the repository root: put the root on TEXINPUTS.
$ENV{'TEXINPUTS'} = '.:..:' . ($ENV{'TEXINPUTS'} // '');
$pdf_mode = 1;
