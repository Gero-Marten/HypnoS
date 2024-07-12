<p align="center">
  <img src="http://outskirts.altervista.org/forum/ext/dmzx/imageupload/img-files/2/ca292f8/8585091/34788e79c6bbe7cf7bb578c6fb4d11f8.jpg">
</p>

<h1 align="center">HypnoS</h1>

## HypnoS Overview


HypnoS is a free and strong UCI chess engine derived from Stockfish 
that analyzes chess positions and computes the optimal moves.

HypnoS does not include a graphical user interface (GUI) that is required 
to display a chessboard and to make it easy to input moves. These GUIs are 
developed independently from HypnoS and are available online.

HypnoS development is currently supported on the Openbench framework. OpenBench (created by Andrew Grant) is an open-source Sequential Probability Ratio Testing (SPRT) framework designed for self-play testing of engines. OpenBench makes use of distributed computing, allowing anyone to contribute CPU time to further the development of some of the world's most powerful engines.

A big thank you goes to the guys at http://chess.grantnet.us/ especially to Andrew Grant for running and improving this testing framework


OpenBench's [Discord server](https://discord.com/invite/9MVg7fBTpM)

## UCI options

The following is a list of options supported by HypnoS (on top of all UCI options supported by Stockfish)
  * #### CTG/BIN Book 1 File
    The file name of the first book file which could be a polyglot (BIN) or Chessbase (CTG) book. To disable this book, use: ```<empty>```
    If the book (CTG or BIN) is in a different directory than the engine executable, then configure the full path of the book file, example:
    ```C:\Path\To\My\Book.ctg``` or ```/home/username/path/to/book/bin```

  * #### Book 1 Width
    The number of moves to consider from the book for the same position. To play best book move, set this option to ```1```. If a value ```n``` (greater than ```1```) is configured, the engine will pick **randomly** one of the top ```n``` moves available in the book for the given position

  * #### Book 1 Depth
    The maximum number of moves to play from the book
	
  * #### (CTG) Book 1 Only Green
    This option is only used if the loaded book is a CTG book. When set to ```true```, the engine will only play Green moves from the book (if any). If no green moves found, then no book move is made
	This option has no effect or use if the loaded book is a Polyglot (BIN) book
    
  * #### CTG/BIN Book 2 File
    Same explanation as **CTG/BIN Book 1 File**, but for the second book

  * #### Book 2 Width
    Same explanation as **Book 1 Width**, but for the second book

  * #### Book 2 Depth
    Same explanation as **Book 1 Depth**, but for the second book

  * #### (CTG) Book 2 Only Green
    Same explanation as **(CTG) Book 1 Only Green**, but for the second book

  * #### Self-Learning

HypnoS implements a persisted learning algorithm, managing a file named experience.bin.

It is a collection of one or more positions stored with the following format (similar to in memory Brainlearn Transposition Table):

- _best move_
- _board signature (hash key)_
- _best move depth_
- _best move score_
- _best move performance_ , a new parameter you can calculate with any learning application supporting this specification.
This file is loaded in an hashtable at the engine load and updated each time the engine receive quit or stop uci command.
When BrainLearn starts a new game or when we have max 8 pieces on the chessboard, the learning is activated and the hash table updated each time the engine has a best score
at a depth >= 4 PLIES, according to HypnoS aspiration window.

At the engine loading, there is an automatic merge to experience.bin files, if we put the other ones, based on the following convention:

&lt;fileType&gt;&lt;qualityIndex&gt;.bin
where

- _fileType=experience_
- _qualityIndex_ , an integer, incrementally from 0 on based on the file&#39;s quality assigned by the user (0 best quality and so on)

N.B.

Because of disk access, less time the engine can think, less effective is the learning.

  * #### Variety
Integer, Default: 0, Min: 0, Max: 40 To play different opening lines from default (0), if not from book (see below).
Higher variety -> more probable loss of ELO

  * #### Variety Max Moves
Adjust how many moves we want HypnoS to use the move variety feature.  

  * #### Options to control engine evaluation strategy
1) Materialistic Evaluation Strategy: Minimum = -12, Maximum = +12, Default = 0. Lower values will cause the engine assign less value to material differences between the sides. More values will cause the engine to assign more value to the material difference.
2) Positional Evaluation Strategy: Minimum = -12, Maximum = +12, Default = 0. Lower values will cause the engine assign less value to positional differences between the sides. More values will cause the engine to assign more value to the positional difference.

The default value for both options (0 = zero) is equivalent to the default evaluation strategy of Stockfish.