# Gene expression inference leveraging chromatin interactions

## ChIA-PET interaction matrix
The class ChiaPetInteractions in src/chiapet_interactions.py allows to convert 
a .bed file of interactions from a ChIA-PET experiment to a contact matrix.<br><br>
It is possible to declare the chromosomes to select by passing the list of chromosomes 
to the "chromosomes" parameter of the ChiaPetInteractions object (e.g. chromosomes=['chr1', 'chrX']).<br>
It is also possible to exclude from the matrix the interactions between regions 
of the same chromosomes, to highlight inter-chromosomes interactions. 
This can be done by passing different_chrs=True when the ChiaPetInteractions object is instantiated.
