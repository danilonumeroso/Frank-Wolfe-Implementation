# Authors
Danilo Numeroso ([@danilonumeroso](https://github.com/danilonumeroso))

Matteo Medioli  ([@Ickarus](https://github.com/Ickarus))

# Implementation
This is a python implementation of the Frank Wolfe algorithm, the goal was clarity rather than efficiency.
This means that it is not optimized (apart from stabilization). Of course
nothing prevents you from doing your own improvements to our implementation.
There's still a lot of work to do to make it faster.

## Dependencies
You'll need either CPlex or ORTools to make it work, as well as pandas and numpy.
Installing ORTools is simpler but it requires to change two lines of code at the end of the file mcfp.py.
You'll need to disable stabilization (=set stabilization to false at line 119 in frank_wolfe.py) too.
Instead, if you want to use CPlex, everything should be ok.

## Usage
```
python frank_wolfe.py <path/to/test> <num arcs> <t>

e.g. python frank_wolfe.py tests/test 1000 15
```
We provided one single test. Anyway, our implementantion is able to read 
[DIMACS](http://archive.dimacs.rutgers.edu/Challenges/) format, therefore you can build your own example.

N.B. frank_wolfe.py assumes <path/to/test> not containing any extensions (neither .dmx nor .qfc)

# License
All source code from this project is released under the MIT license.
