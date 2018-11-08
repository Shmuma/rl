# 2018-11-08

MCTS performance drops with increase of c:
````
(art_01_cube) shmuma@gpu:~/work/rl/articles/01_rubic$ ./solver.py -e cube2x2 -m saves/cube2x2-zero-goal-d200-t1/best_1.4547e-02.dat --max-steps 1000 --cuda -r 20
2018-11-08 06:33:56,195 INFO Using environment CubeEnv('cube2x2')
2018-11-08 06:33:58,169 INFO Network loaded from saves/cube2x2-zero-goal-d200-t1/best_1.4547e-02.dat
2018-11-08 06:33:58,169 INFO Got task [10, 1, 0, 11, 4, 3, 3, 2, 11, 1, 10, 11, 8, 1, 9, 6, 1, 3, 3, 8], solving...
2018-11-08 06:34:01,330 INFO Maximum amount of steps has reached, cube wasn't solved. Did 1001 searches, speed 316.77 searches/s
````

* c=10k: 316 searches/s
* c=100k: 58 searches/s
* c=1m: 4.94 searches/s

Root tree state is the same for 10k and 100k.

Mean search depth: 1k: 57, 10k: 129.7, 100k: 861

Conclusion:
Larger C makes tree taller by exploring less options around, but delving deeper into the search space.
This leads to longer search paths which take more and more time to back up.
It is likely that my C value is too large and I just need to speed up MCTS.

TODO: 
* measure depths of resulting tree
* analyze the length of solution (both naive and BFS)
* check effect of C on those parameters

Depths with 1000 steps limit:
* c=1m:   {'min': 1, 'max': 16, 'mean': 7.849963731321631, 'leaves': 34465}
* c=100k: {'min': 1, 'max': 17, 'mean': 9.103493774652236, 'leaves': 71241}
* c=10k:  {'min': 1, 'max': 18, 'mean': 10.28626504647809, 'leaves': 70033}
* c=1k:   {'min': 1, 'max': 18, 'mean': 9.942448384493218, 'leaves': 76818}
* c=100:  {'min': 1, 'max': 14, 'mean': 8.938883245826121, 'leaves': 69899}
* c=10:   {'min': 1, 'max': 13, 'mean': 8.59500956472128,  'leaves': 59594}

Depths with 10000 steps limit:
* c=10k:  {'min': 1, 'max': 27, 'mean': 15.374430775030191, 'leaves': 1289253}
* c=1k:   {'min': 1, 'max': 26, 'mean': 14.057022074409328, 'leaves': 1004874}
* c=100:  {'min': 1, 'max': 19, 'mean': 12.376234716455224, 'leaves': 1113616}
* c=10:   {'min': 1, 'max': 19, 'mean': 11.707333613164712, 'leaves': 886248}

c=100 with different limits:
* 100k: 
